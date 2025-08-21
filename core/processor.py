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
            for sample in df.head(2).to_dict(orient='records'):
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

重要提示：
0. 处理数据时，用户提示词输入的内容不一定能在数据中直接找到对应列，遇到这种情况请自行决定
1. 返回的内容只能是可直接执行的代码
2. 不要有任何对代码的说明或者其他说明
3. 保证返回的内容可以直接执行
4. 不能作为代码执行的内容放在summary字符串中
5. 代码中使用变量或函数时先进行定义，确保代码可以直接执行
6. 生成图表时必须确保使用的列存在于数据中
7. 调用argmax()、idxmax()、argmin()、idxmin()等函数前，必须先检查数据是否为空：
   - 示例：若需计算df['col'].idxmax()，需先判断`if not df['col'].empty and df['col'].notna().any()`
   - 若数据为空，需在summary中说明"因数据为空，无法计算最大值/最小值"，避免报错
8. 处理分组数据（如groupby后）时，需检查每个分组是否为空，再进行聚合操作
说明：
0. 严格保证代码语法与库方法使用正确性：
   - 所有括号（包括圆括号、方括号、花括号）必须严格匹配，确保每个左括号都有对应的右括号，且嵌套关系正确
   - 检查所有语句的语法完整性，避免出现未闭合的括号、引号等情况
   - 所有Pandas方法（如groupby、reset_index、rename等）必须使用官方支持的参数，禁止使用不存在的参数（如reset_index的'name'参数）
   - 处理DataFrame时：
   - 在处理 pandas 数据框时，请注意：
     (1)计算最大值/最小值位置时，`idxmax()`/`idxmin()` 返回的是行索引（整数），而非行数据本身
     (2)正确步骤应为：先通过 idxmax() 获取索引 → 再用 loc[索引] 获取行数据
     (3)禁止直接使用 `df.loc[df['列名'].idxmax()]` 的结果作为新的索引值
     (4)数值列（如计数、数量）应保持整数/浮点类型，避免转为字符串导致计算错误
     (5)在生成总结文字时，需先获取完整行数据，再提取其中的字段值（如时段和次数）
   - 特别注意：reset_index()方法不支持'name'参数，如需重命名列请使用rename()或在groupby时指定
   - 方法参数名称必须准确无误，禁止使用不存在的参数
   - 检查方法调用的参数数量与类型是否匹配
1. 已存在变量data_dict（文件名到DataFrame的字典），可直接使用
2. 必须导入所需的库（如pandas、datetime）
3. 必须定义三个变量：
   - result_table：处理后的DataFrame结果（必须存在）
   - summary：字符串类型的总结，根据用户要求生成具体内容
     * 关键分析结论（如统计数量、趋势、异常点等）
     * 数据中发现的规律总结
     * 针对问题的解决方案或建议
     * 其他用户要求但无法被作为代码执行的信息
     禁止使用默认值，必须根据分析结果生成具体内容
   - chart_info：图表信息字典（** 必须包含chart_type字段 **），格式为:
     {{
       "chart_type": "bar/line/pie/scatter/hist",  # 强制必填，且为支持的类型
       "title": "图表标题",  # 强制必填
       "data_prep": {{
         "x_col": "x轴数据列名",  # bar/line/scatter/hist必须提供
         "y_col": "y轴数据列名",  # bar/line/scatter可选
         "values": "值列名",  # pie必须提供
         "bins": 分箱数  # hist可选
       }}
     }}
     这是图表信息的模板，chart_type字段按照这些关键词对应：bar柱状图/line折线图/scatter散点图/hist直方图/pie饼图
     生成代码时根据图表类型检查必要的列配置
            'bar': ['x_col', 'y_col']
            'line': ['x_col', 'y_col']
            'scatter': ['x_col', 'y_col']
            'pie': ['x_col', 'values']
            'hist': ['x_col']
     当用户说生成“图表”这类泛指时，由你决定图表类型（即chart_type字段）
     若不需要图表，chart_info必须显式设置为None；若需要图表，所有标注"必须"的字段均为必填项
     在生成图表信息时，需严格遵循以下规范：
     (1)列名验证前置：生成`chart_info`前，必须先获取`result_table`的所有列名（通过`result_table.columns.tolist()`），并确认所有指定的列（如`x_col`、`y_col`、`values`）均在列名列表中，禁止使用数据中不存在的列名。
     (2)动态适配列名：若用户需求中提到的列名（如“次数”）与`result_table`实际列名不一致，需使用实际列名，并在`summary`中说明列名对应关系（例如：“注：图表中使用的‘count’列对应需求中的‘次数’”）。
     (3)错误规避处理：若用户需求中的列名不存在且无替代列，应放弃生成该图表，将`chart_info`设为`None`，并在`summary`中说明原因（例如：“因数据中无‘次数’列，无法生成对应图表”）。
     (4)显式列名引用：在`chart_info`的`data_prep`中，所有列名必须直接引用`result_table`中存在的列，禁止使用任何推测性列名或别名。
4. 处理时间/日期类型列（如包含timestamp、datetime的列）：
   - 必须显式转换为字符串类型（如df['time'] = df['time'].astype(str)或df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')）
   - 确保无任何Timestamp类型数据残留，避免JSON序列化错误
5. 处理含中文的列（如"低/中/高"）：显式转换为字符串类型（df['level'] = df['level'].astype(str)）
6. 对同义表头进行统一命名和内容整合
7. 进行列重命名时，确保新列名的数量与 DataFrame 实际列数完全一致
8. 若需重置索引，优先使用 drop=True 参数（如 reset_index (drop=True)）避免引入多余的索引列
强制语法检查清单：
1. 括号完整性：
   - 所有圆括号、方括号、花括号必须成对出现，嵌套关系正确（如函数参数括号、列表括号）。
   - 反面示例：`pd.concat([df1, axis=0)`（缺少右括号和第二个DataFrame参数）是绝对禁止的。
2. 参数完整性：
   - 函数调用必须包含必要参数（如 `pd.concat` 必须传入包含至少一个DataFrame的列表作为第一个参数）。
   - 反面示例：`pd.concat([df1])` 虽然语法正确，但需确认是否遗漏其他需合并的DataFrame（如 `df2`）。
3. 语句终结符：
   - 每行代码末尾无多余逗号/括号（如 `df['col'] = 1,` 会导致语法错误）。
4. 变量引用有效性：
   - 确保所有引用的变量已定义（如合并时使用的 `df2` 必须在之前被正确定义为 `data_dict` 中的数据）。
生成后自检流程
1. 逐行检查代码是否存在上述语法错误（重点检查括号、逗号、参数数量）。
2. 对 `pd.concat`、`pd.groupby` 等高频错误函数，额外确认参数格式（如 `pd.concat` 的第一个参数必须是列表 `[df1, df2]`）。
3. 假设自己是Python解释器，模拟执行前3行代码，确认无语法报错。
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
        """直接回答模式：生成日志总结，不返回表格数据"""
        data_dict = self._load_file_data(file_names)

        # 收集文件详细信息（包含敏感词处理）
        file_details = []
        for filename, df in data_dict.items():
            # 1. 处理列名中的敏感词
            processed_columns = [
                self.sensitive_processor.normalize_to_replacement(col)
                for col in df.columns.tolist()
            ]

            # 2. 处理数据样本中的敏感词
            processed_samples = []
            for sample in df.head(min(3, len(df))).to_dict(orient='records'):
                processed_sample = {}
                for key, value in sample.items():
                    if isinstance(value, str):
                        processed_val = self.sensitive_processor.normalize_to_replacement(value)
                        processed_sample[key] = processed_val
                    else:
                        processed_sample[key] = value
                processed_samples.append(processed_sample)

            # 基础信息（使用处理后的列名和样本）
            details = {
                "文件名": filename,
                "记录数": len(df),
                "列名": processed_columns,
                "数据类型分布": {col: str(df[col].dtype) for col in df.columns},
                "数据样本": processed_samples  # 最多3行样本
            }

            # 数值列统计（无需处理敏感词）
            numeric_stats = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_stats[col] = {
                        "平均值": df[col].mean(),
                        "最小值": df[col].min(),
                        "最大值": df[col].max(),
                        "非空值数量": df[col].count()
                    }
            if numeric_stats:
                details["数值列统计"] = numeric_stats

            file_details.append(details)

        # 处理用户请求中的敏感词
        processed_request = self.sensitive_processor.normalize_to_replacement(user_request)

        # 构建提示词（使用处理后的请求和文件详情）
        prompt = f"""基于以下日志文件的详细信息，回答用户问题并生成总结:
    文件详情: {json.dumps(file_details, ensure_ascii=False, default=str)}
    用户问题: {processed_request}

    回答要求:
    1. 深入分析日志数据特征、潜在规律和关键信息
    2. 直接给出自然语言总结，不生成任何表格或结构化数据
    3. 内容具体有针对性，避免泛泛而谈
    4. 涉及统计信息时自然体现关键数值
    5. 用简洁易懂的中文表达"""

        # 调用API
        response = self.client.completions_create(
            model='deepseek-reasoner',
            prompt=prompt,
            max_tokens=5000,
            temperature=1.0
        )

        # 对返回结果进行敏感词还原（关键修复：之前缺少这一步）
        restored_summary = self.sensitive_processor.restore_sensitive_words(
            response.choices[0].message.content.strip()
        )

        return {"summary": restored_summary}