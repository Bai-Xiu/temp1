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
        """执行代码并简化图表配置校验，使用预加载的数据"""
        full_code = f"{cleaned_code}\n"
        local_vars = {
            'data_dict': data_dict,
            'pd': pd,
            'np': np
        }

        try:
            exec(full_code, globals(), local_vars)
            result_table = local_vars.get('result_table')
            # 强制获取总结并进行敏感词还原（确保代码生成模式下必然执行）
            summary = local_vars.get('summary', '分析完成但未生成总结')
            # 关键修改：无论何种情况都对总结执行还原处理
            summary = self.processor.sensitive_processor.restore_sensitive_words(summary)

            chart_info = local_vars.get('chart_info', None)

            # 还原表格数据（处理字符串列）
            if result_table is not None and isinstance(result_table, pd.DataFrame):
                for col in result_table.columns:
                    if result_table[col].dtype == 'object':
                        result_table[col] = result_table[col].apply(
                            lambda x: self.processor.sensitive_processor.restore_sensitive_words(str(x)) if pd.notna(
                                x) else x
                        )

            # 还原图表信息中的文本
            if chart_info and isinstance(chart_info, dict):
                if 'title' in chart_info:
                    chart_info['title'] = self.processor.sensitive_processor.restore_sensitive_words(
                        chart_info['title'])
                if 'data_prep' in chart_info and isinstance(chart_info['data_prep'], dict):
                    for key, value in chart_info['data_prep'].items():
                        if isinstance(value, str):
                            chart_info['data_prep'][key] = self.processor.sensitive_processor.restore_sensitive_words(
                                value)

            # 图表配置校验警告
            if chart_info and isinstance(chart_info, dict):
                top_required = ["chart_type", "title", "data_prep"]
                missing_top = [f for f in top_required if f not in chart_info]
                if missing_top:
                    summary += f"\n警告：图表配置缺少顶级字段 {missing_top}"
                data_prep = chart_info.get("data_prep", {})
                if not isinstance(data_prep, dict):
                    summary += "\n警告：data_prep必须是字典类型"
                    chart_info["data_prep"] = {}

            if result_table is None:
                result_table = pd.concat(data_dict.values(), ignore_index=True)
                summary = "未生成有效分析结果，返回原始数据合并表格\n" + summary

            return {
                "result_table": result_table,
                "summary": summary,
                "chart_info": chart_info
            }
        except Exception as e:
            # 对错误信息也进行敏感词还原
            error_msg = f"代码执行错误: {str(e)}\n\n执行的代码:\n{full_code}"
            return {
                "summary": self.processor.sensitive_processor.restore_sensitive_words(error_msg),
                "result_table": None,
                "chart_info": None
            }