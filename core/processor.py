import os
import pandas as pd
import json
from utils.helpers import get_file_list, sanitize_filename
from core.api_client import DeepSeekAPI
from core.file_processors import (
    CsvFileProcessor, ExcelFileProcessor,
    JsonFileProcessor, TxtFileProcessor
)


class LogAIProcessor:
    def __init__(self, config):
        self.config = config
        self.api_key = config.get("api_key", "")

        # 添加敏感词处理器
        from core.sensitive_processor import SensitiveWordProcessor
        self.sensitive_processor = SensitiveWordProcessor(config)

        # 区分默认目录和当前目录
        self.default_data_dir = config.get("data_dir", "")  # 持久化的默认目录
        self.current_data_dir = self.default_data_dir  # 当前工作目录（临时）

        self.default_save_dir = config.get("save_dir", "")  # 持久化的默认目录
        self.current_save_dir = self.default_save_dir  # 当前工作目录（临时）

        self.verbose = config.get("verbose_logging", False)
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312', 'ansi', 'utf-16', 'utf-16-le']

        # 初始化API客户端，传入敏感词处理器
        self.client = DeepSeekAPI(api_key=self.api_key,
                                  sensitive_processor=self.sensitive_processor) if self.api_key else None

        # 存储当前选择的文件和数据
        self.current_files = None
        self.current_data = None

        # 初始化文件处理器（核心扩展点：添加新类型只需在这里注册）
        self.file_processors = [
            CsvFileProcessor(),
            ExcelFileProcessor(),
            JsonFileProcessor(),
            TxtFileProcessor()
        ]

        # 构建扩展名到处理器的映射
        self.extension_map = {}
        for processor in self.file_processors:
            for ext in processor.get_supported_extensions():
                self.extension_map[ext.lower()] = processor

    def set_default_data_dir(self, new_dir):
        if new_dir:
            self.default_data_dir = new_dir
            self.config.set("data_dir", new_dir)

    def set_current_data_dir(self, new_dir):
        if new_dir:
            self.current_data_dir = new_dir

    def set_default_save_dir(self, new_dir):
        if new_dir:
            self.default_save_dir = new_dir
            self.config.set("save_dir", new_dir)

    def set_current_save_dir(self, new_dir):
        if new_dir:
            self.current_save_dir = new_dir

    def get_file_list(self):
        """获取当前数据目录中的文件列表"""
        if not self.current_data_dir or not os.path.exists(self.current_data_dir):
            return []
        return get_file_list(self.current_data_dir)

    def load_data_files(self, file_names):
        """从当前数据目录加载文件"""
        if not self.current_data_dir or not os.path.exists(self.current_data_dir):
            raise ValueError("当前数据目录未设置或不存在")

        return self._load_file_data(file_names)

    def _load_file_data(self, file_names):
        """从当前数据目录读取文件数据，使用多线程加载"""
        if self.current_data and set(file_names) == set(self.current_data.keys()):
            return self.current_data

        data_dict = {}
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def load_single_file(file_name):
            safe_file = sanitize_filename(file_name)
            full_path = os.path.join(self.current_data_dir, safe_file)

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"文件不存在: {full_path}")

            _, ext = os.path.splitext(full_path)
            ext = ext.lower()

            if ext not in self.extension_map:
                supported_exts = ", ".join(self.extension_map.keys())
                raise ValueError(f"不支持的文件格式: {ext}。支持的格式: {supported_exts}")

            processor = self.extension_map[ext]
            df = processor.read_file(full_path, encodings=self.supported_encodings)

            # 加载后立即进行敏感词处理
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].astype(str).apply(
                        lambda x: self.sensitive_processor.normalize_to_replacement(x) if pd.notna(x) else x
                    )
            return safe_file, df_copy

        with ThreadPoolExecutor(max_workers=min(4, len(file_names))) as executor:
            futures = {executor.submit(load_single_file, fn): fn for fn in file_names}

            for future in as_completed(futures):
                try:
                    safe_file, df = future.result()
                    data_dict[safe_file] = df
                except Exception as e:
                    raise RuntimeError(f"读取文件 {futures[future]} 失败: {str(e)}")

        self.current_data = data_dict
        return data_dict

    def process_and_anonymize_files(self, file_names, output_dir):
        """处理并去敏文件"""
        if not file_names:
            raise ValueError("未选择文件")

        if not output_dir or not os.path.exists(output_dir):
            raise ValueError("无效的输出目录")

        data_dict = self._load_file_data(file_names)
        results = {}

        for filename, df in data_dict.items():
            # 对DataFrame中的文本进行去敏处理
            anonymized_df = self._anonymize_dataframe(df)

            # 保存去敏后的文件
            base_name = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            output_path = os.path.join(
                output_dir,
                f"{base_name}_anonymized{ext}"
            )

            # 根据文件类型保存
            _, ext = os.path.splitext(filename)
            if ext.lower() in ['.csv']:
                anonymized_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            elif ext.lower() in ['.xlsx', '.xls']:
                anonymized_df.to_excel(output_path, index=False)
            elif ext.lower() in ['.json']:
                anonymized_df.to_json(output_path, orient='records', force_ascii=False)
            else:  # 文本文件
                content = "\n".join(anonymized_df.iloc[:, 0].astype(str).tolist())
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            results[filename] = output_path

        return results

    def _anonymize_dataframe(self, df):
        """对DataFrame进行去敏处理，使用向量化操作"""
        df_copy = df.copy()

        # 对每一列进行处理
        for col in df_copy.columns:
            # 处理字符串类型的列，使用向量化操作
            if df_copy[col].dtype == 'object':
                # 使用applymap代替apply提高性能
                df_copy[col] = df_copy[col].astype(str).apply(
                    lambda x: self._anonymize_text(x) if pd.notna(x) else x
                )

        return df_copy

    def _anonymize_text(self, text):
        """对文本进行去敏处理"""
        if not text or not isinstance(text, str):
            return text

        # 使用敏感词处理器进行替换
        anonymized_text, _ = self.sensitive_processor.replace_sensitive_words(text)
        return anonymized_text

    def generate_processing_code(self, user_request, file_names):
        """生成完整可执行代码，而非函数内部逻辑"""
        if not self.client:
            # 默认代码：直接返回所有数据
            return """import pandas as pd
    result_table = pd.concat(data_dict.values(), ignore_index=True)
    # 转换所有时间类型列
    for col in result_table.columns:
        if pd.api.types.is_datetime64_any_dtype(result_table[col]):
            result_table[col] = result_table[col].astype(str)
    summary = f'共{len(result_table)}条记录'
    chart_info = None"""

        data_dict = self._load_file_data(file_names)
        file_info = {}

        # 处理用户请求，确保其中的敏感词被统一替换
        processed_request = self.sensitive_processor.normalize_to_replacement(user_request)

        for filename, df in data_dict.items():
            # 1. 替换列名中的敏感词
            replaced_columns = [
                self.sensitive_processor.normalize_to_replacement(col)
                for col in df.columns.tolist()
            ]

            # 2. 替换样本数据中的敏感词
            replaced_samples = []
            for sample in df.head(5).to_dict(orient='records'):
                replaced_sample = {}
                for key, value in sample.items():
                    # 对每个字段值进行替换（处理字符串类型）
                    if isinstance(value, str):
                        replaced_val = self.sensitive_processor.normalize_to_replacement(value)
                        replaced_sample[key] = replaced_val
                    else:
                        replaced_sample[key] = value  # 非字符串类型保持不变
                replaced_samples.append(replaced_sample)

            file_info[filename] = {
                "columns": replaced_columns,
                "sample": replaced_samples
            }

        # 3. 生成代码
        prompt = f"""根据用户请求编写完整的Python处理代码:
用户需求: {user_request}
数据信息: {json.dumps(file_info, ensure_ascii=False)}


请根据用户需求生成日志分析代码，生成代码时必须严格遵循以下与项目代码适配的规范，确保代码可直接执行且符合项目逻辑：

### 一、数据源与基础环境说明
1. **变量定义强制约束**：
   - 禁止使用df变量名，需使用`result_table`变量名，避免与项目代码中的DataFrame变量冲突。
   - 所有使用的变量必须先定义再使用，尤其是数据合并后的主数据变量，需在所有代码路径中确保初始化。
   - 数据合并必须基于`data_dict`

2. **分支逻辑完整性**：
   - 所有条件判断（如`if-else`）必须覆盖完整场景，确保`df`在任何分支中都有定义。
   - 对`pd.concat`等可能抛出异常的操作，必须包裹`try-except`块，异常分支中需显式定义`df = pd.DataFrame()`。

3. **代码可执行性**：
   - 禁止引用未定义的变量、函数或模块，需提前导入`pandas`（`import pandas as pd`）。
   - 生成的代码必须包含完整的变量初始化、数据处理、结果输出（`result_table`、`summary`、`chart_info`）逻辑，可直接在包含`data_dict`的环境中运行。

4. **敏感词处理兼容**：
   - 代码中涉及的列名、文本处理需兼容`PROTECTEDxxxx`格式的敏感词替换结果，不影响变量定义和逻辑执行。

### 二、必须定义的核心变量规范
1. `result_table`：类型为`pandas.DataFrame`（必须存在），存储处理后的核心数据
   - 时间类型列（`datetime64`）需显式转换为字符串（如`df['time'] = df['time'].astype(str)`）
   - 含中文的列需显式转换为字符串类型（如`df['level'] = df['level'].astype(str)`）
2. `summary`：字符串类型，存储分析结论，需包含：
   - 关键分析结果（统计数量、趋势等）
   - 数据规律总结
   - 异常情况说明（如空数据、缺失列）
   - 禁止使用默认值，必须基于实际分析生成
   - 涉及敏感词需保留`PROTECTEDXXXXXXXX`格式，不得使用引用
3. `chart_info`：图表信息字典（或`None`），必须符合以下格式：
   {{
     "chart_type": "bar/line/pie/scatter/hist",  # 仅支持这5种类型
     "title": "图表标题",  # 强制必填
     "data_prep": {{
       "x_col": "x轴数据列名",  # bar/line/scatter/hist必填
       "y_col": "y轴数据列名",  # bar/line/scatter必填
       "values": "值列名",     # pie必填
       "bins": 分箱数          # hist可选，默认10
     }}
   }}
   - 生成前必须验证`x_col`/`y_col`/`values`对应的列存在于`result_table.columns`中
   - 若列名与用户需求不一致，需在`summary`中说明对应关系
   - 若所需列不存在，需将`chart_info`设为`None`并在`summary`说明原因

### 三、数据处理强制规则
1. 多文件合并：必须先通过`df = pd.concat(data_dict.values(), ignore_index=True)`合并数据为`df`后再处理
2. 列操作前置检查：访问列前必须用`col in df.columns`验证，避免KeyError
3. 空数据处理：
   - 调用`unique()`/`idxmax()`/`idxmin()`/`argmax()`/`argmin()`前，必须检查：
     `if isinstance(obj, pd.Series) and not obj.empty and obj.notna().any()`
   - 空数据场景需在`summary`中说明（如"因数据为空，无法计算最大值"）
4. 分组数据处理：`groupby`后需检查每个分组是否为空再进行聚合操作
5. 索引重置：优先使用`reset_index(drop=True)`避免引入多余索引列
6. 同义表头处理：需统一命名并整合内容，重命名时确保新列名数量与实际列数一致

### 四、代码语法与执行规范
1. 仅返回可直接执行的代码，无任何说明文字
2. 变量/函数使用前必须定义，确保无未定义错误
3. 处理可能的异常（AttributeError/KeyError等），并在`summary`中提示具体错误原因
4. 严格检查括号完整性（圆括号/方括号/花括号必须成对，嵌套正确）
5. 函数调用必须包含必要参数（如`pd.concat`需传入DataFrame列表）
6. 语句末尾无多余逗号/括号，避免语法错误
7. Pandas方法使用官方支持参数（如`reset_index`不支持`name`参数，重命名用`rename()`）
8. 数值列保持整数/浮点类型，避免转为字符串导致计算错误

请基于上述规范，根据用户的日志分析需求生成代码，确保代码符合项目代码逻辑且可直接执行。
"""

        response = self.client.completions_create(
            model='deepseek-reasoner',
            prompt=prompt,
            max_tokens=10000,
            temperature=1.0
        )

        code_block = response.choices[0].message.content.strip()

        return code_block

    def direct_answer(self, user_request, file_names):
        """直接回答模式：对全部数据脱敏后调用API，不展示表格内容"""
        if not self.client:
            return {
                "summary": "未配置API密钥，无法进行直接回答",
                "result_table": None,  # 直接回答模式不展示表格
                "chart_info": None
            }

        # 加载数据并脱敏（处理全部数据）
        data_dict = self._load_file_data(file_names)
        file_info = {}
        for filename, df in data_dict.items():
            # 1. 脱敏列名
            replaced_columns = [
                self.sensitive_processor.replace_sensitive_words(col)[0]
                for col in df.columns.tolist()
            ]

            # 2. 脱敏全部数据（不再只传样本，而是全部记录）
            replaced_records = []
            for record in df.to_dict(orient='records'):  # 遍历所有行，而非head(2)
                replaced_record = {}
                for key, value in record.items():
                    if isinstance(value, str):
                        # 对字符串类型脱敏
                        replaced_val, _ = self.sensitive_processor.replace_sensitive_words(value)
                        replaced_record[key] = replaced_val
                    else:
                        replaced_record[key] = value  # 非字符串保持原样
                replaced_records.append(replaced_record)

            file_info[filename] = {
                "columns": replaced_columns,
                "records": replaced_records  # 用全部记录替换样本
            }

        # 构建脱敏后的prompt（明确基于全部数据回答）
        prompt = f"""用户需求: {user_request}
    数据信息: {json.dumps(file_info, ensure_ascii=False)}

    请基于提供的全部数据信息回答用户问题，无需生成代码。
    """

        # 调用API（仅传递脱敏数据）
        response = self.client.completions_create(prompt=prompt)
        raw_summary = response.choices[0].message.content

        # 本地还原总结内容（关键修改：确保在本地完成还原）
        restored_summary = self.sensitive_processor.restore_sensitive_words(raw_summary)

        return {
            "summary": restored_summary,  # 返回还原后的总结
            "result_table": None,  # 直接回答模式不返回表格
            "chart_info": None
        }