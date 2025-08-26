import re
import pandas as pd
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class AnalysisThread(QThread):
    update_signal = pyqtSignal(str)
    complete_signal = pyqtSignal(dict)

    def __init__(self, processor, file_paths, request, mode):
        super().__init__()
        self.processor = processor
        self.file_paths = file_paths
        self.request = request
        self.mode = mode

    def run(self):
        try:
            self.update_signal.emit("正在加载数据...")
            data_dict = self.processor.load_data_files(self.file_paths)
            self.update_signal.emit("数据加载完成，正在进行分析...")

            if self.mode == "1":
                # 代码处理模式：生成代码并执行
                code_block = self.processor.generate_processing_code(self.request, self.file_paths)
                self.update_signal.emit("代码生成完成，开始执行...")
                cleaned_code = self.clean_code_block(code_block)
                result = self.execute_cleaned_code(cleaned_code, data_dict)

                # 关键补充：代码模式下也对总结进行本地还原
                if "summary" in result and result["summary"]:
                    result["summary"] = self.processor.sensitive_processor.restore_sensitive_words(result["summary"])
            else:
                # 直接回答模式（使用上面修改后的方法）
                result = self.processor.direct_answer(self.request, self.file_paths)

            self.complete_signal.emit({"status": "success", "result": result})
        except Exception as e:
            self.complete_signal.emit({"status": "error", "message": str(e)})

    def clean_code_block(self, code_block):
        """清理代码块，移除三重反引号和语言标识"""
        if not code_block:
            return ""
        cleaned = re.sub(r'^```[\w]*', '', code_block, flags=re.MULTILINE)
        cleaned = re.sub(r'```$', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def execute_cleaned_code(self, cleaned_code, data_dict):
        """执行代码并增强鲁棒性，确保核心变量始终被定义"""
        # 1. 预初始化所有可能用到的变量（包括df），设置默认值
        local_vars = {
            'data_dict': data_dict,
            'pd': pd,
            'np': np,
            # 强制初始化必须返回的变量
            'result_table': pd.DataFrame(),
            'summary': '分析完成但未生成有效总结',
            'chart_info': None,
            # 关键：提前初始化df，避免AI生成代码遗漏定义
            'df': pd.DataFrame()  # 新增df的默认初始化
        }

        try:
            # 2. 安全检查代码（去除多余缩进，确保语法正确）
            safety_check_code = """
# 生成代码执行前的安全检查
if not isinstance(data_dict, dict):
    raise ValueError("data_dict必须是字典类型")
for key, val in data_dict.items():
    if not isinstance(val, pd.DataFrame):
        raise TypeError(f"数据 {key} 不是DataFrame类型")
    """
            full_code = f"{safety_check_code}\n{cleaned_code}"
            exec(full_code, globals(), local_vars)

            # 3. 提取变量时再次校验类型
            result_table = local_vars.get('result_table')
            if not isinstance(result_table, pd.DataFrame):
                result_table = pd.DataFrame()
                local_vars['summary'] += "\n警告：result_table类型异常，已自动修正为空表"

            # 4. 敏感词还原（适配表格中PROTECTE开头的敏感词）
            summary = local_vars.get('summary', '分析完成但未生成总结')
            summary = self.processor.sensitive_processor.restore_sensitive_words(summary)

            # 还原表格中的敏感词（针对所有字符串列）
            if isinstance(result_table, pd.DataFrame) and not result_table.empty:
                for col in result_table.columns:
                    if result_table[col].dtype == 'object':
                        result_table[col] = result_table[col].apply(
                            lambda x: self.processor.sensitive_processor.restore_sensitive_words(str(x))
                            if pd.notna(x) else x
                        )

            # 处理图表信息
            chart_info = local_vars.get('chart_info', None)
            if isinstance(chart_info, dict):
                # 还原图表标题中的敏感词
                if 'title' in chart_info:
                    chart_info['title'] = self.processor.sensitive_processor.restore_sensitive_words(
                        chart_info['title'])

            return {
                "result_table": result_table,
                "summary": summary,
                "chart_info": chart_info
            }

        except Exception as e:
            # 错误提示更精准，结合表格特点
            error_msg = (
                f"代码执行错误: {str(e)}\n"
                f"错误位置提示: 可能是敏感词处理冲突或df初始化异常\n"
                f"当前表格特征: 包含PROTECTE开头的敏感词替换字段，共{len(data_dict)}个数据表\n"
                f"执行的代码:\n{full_code}"
            )
            return {
                "summary": self.processor.sensitive_processor.restore_sensitive_words(error_msg),
                "result_table": pd.DataFrame(),
                "chart_info": None
            }