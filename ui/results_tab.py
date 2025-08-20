from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem,
                            QSplitter, QFileDialog, QButtonGroup)
from PyQt5.QtCore import Qt
import os
import pandas as pd
from utils.helpers import show_info_message, show_error_message, get_unique_filename
from ui.charts_widget import ChartsWidget  # 导入图表组件
from utils.plot_helpers import prepare_chart_data  # 导入数据处理函数（修改了导入路径）
from PyQt5.QtCore import pyqtSignal, QObject


class ResultsTab(QWidget):
    plot_chart_signal = pyqtSignal(dict, str, str, str, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.parent = parent
        self.current_result = None
        self.current_save_dir = config.get("save_dir")
        self.init_ui()
        self.plot_chart_signal.connect(self._plot_chart_main_thread)

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 保存路径选择（保持不变）
        save_path_layout = QHBoxLayout()
        save_path_layout.addWidget(QLabel("保存路径:"))

        self.save_dir_edit = QLineEdit(self.current_save_dir)
        self.save_dir_edit.setReadOnly(False)
        save_path_layout.addWidget(self.save_dir_edit)

        self.change_save_dir_btn = QPushButton("浏览...")
        self.change_save_dir_btn.clicked.connect(self.change_save_dir)
        save_path_layout.addWidget(self.change_save_dir_btn)

        self.apply_save_dir_btn = QPushButton("应用")
        self.apply_save_dir_btn.clicked.connect(self.apply_save_dir)
        save_path_layout.addWidget(self.apply_save_dir_btn)

        # 分割器
        splitter = QSplitter(Qt.Vertical)

        # 总结区域（保持不变）
        summary_group = QGroupBox("分析总结")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_display = QTextEdit()
        self.summary_display.setReadOnly(True)
        summary_layout.addWidget(self.summary_display)
        splitter.addWidget(summary_group)

        # 结果展示区域（修改部分）
        result_group = QGroupBox("结果展示")
        result_layout = QVBoxLayout(result_group)

        # 切换按钮
        self.view_buttons = QButtonGroup(self)
        self.table_btn = QPushButton("结果表格")
        self.chart_btn = QPushButton("结果图表")

        # 设置按钮为互斥
        self.view_buttons.addButton(self.table_btn, 1)
        self.view_buttons.addButton(self.chart_btn, 2)
        self.table_btn.setCheckable(True)
        self.chart_btn.setCheckable(True)
        self.table_btn.setChecked(True)  # 默认表格视图

        # 绑定事件
        self.table_btn.clicked.connect(self.show_table)
        self.chart_btn.clicked.connect(self.show_chart)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.table_btn)
        btn_layout.addWidget(self.chart_btn)
        btn_layout.addStretch()

        # 结果展示容器
        self.result_container = QWidget()
        self.result_container_layout = QVBoxLayout(self.result_container)

        # 表格组件
        self.result_table = QTableWidget()

        # 图表组件
        self.chart_widget = ChartsWidget()
        self.chart_widget.setVisible(False)  # 初始隐藏图表

        # 添加到容器
        self.result_container_layout.addWidget(self.result_table)
        self.result_container_layout.addWidget(self.chart_widget)

        # 添加到结果布局
        result_layout.addLayout(btn_layout)
        result_layout.addWidget(self.result_container)

        splitter.addWidget(result_group)
        splitter.setSizes([200, 400])

        # 按钮区（保持不变）
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)

        self.new_analysis_btn = QPushButton("新分析")
        self.new_analysis_btn.clicked.connect(self.start_new_analysis)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.new_analysis_btn)

        layout.addLayout(save_path_layout)
        layout.addWidget(splitter)
        layout.addLayout(btn_layout)

    def set_result(self, result):
        """更新结果展示，同时准备图表数据"""
        self.current_result = result
        self.display_results(result)

        # 检查是否有图表信息，如果有则准备图表数据
        try:
            if ("chart_info" in result and result["chart_info"] is not None and
                    isinstance(result["chart_info"], dict) and
                    "result_table" in result and isinstance(result["result_table"], pd.DataFrame)):

                # 启用图表按钮
                self.chart_btn.setEnabled(True)

                # 准备图表数据
                # 修改为：传递完整的 chart_info 字典
                chart_data = prepare_chart_data(
                    result["result_table"],
                    result["chart_info"]
                )

                # 如果图表数据有效，绘制图表
                if chart_data:
                    self.chart_widget.plot_chart(
                        chart_data,
                        result["chart_info"]["chart_type"],  # 直接从顶级获取
                        result["chart_info"]["title"],
                        x_label=result["chart_info"].get("x_col"),  # 直接从顶级获取
                        y_label=result["chart_info"].get("y_col")
                    )
                else:
                    self.chart_btn.setEnabled(False)
            else:
                # 禁用图表按钮
                self.chart_btn.setEnabled(False)
        except Exception as e:
            show_error_message(self, "图表处理错误", f"处理图表数据时出错: {str(e)}")
            self.chart_btn.setEnabled(False)

    def _plot_chart_main_thread(self, data, chart_type, title, x_label, y_label):
        """主线程中实际执行图表绘制"""
        self.chart_widget.plot_chart(data, chart_type, title, x_label, y_label)

    def display_results(self, result):
        """显示结果表格"""
        # 显示总结
        if "summary" in result:
            self.summary_display.setText(result["summary"])
            self.save_btn.setEnabled("result_table" in result and result["result_table"] is not None)

        # 显示表格
        if "result_table" in result and isinstance(result["result_table"], pd.DataFrame):
            df = result["result_table"].copy()
            self.result_table.setRowCount(df.shape[0])
            self.result_table.setColumnCount(df.shape[1])
            self.result_table.setHorizontalHeaderLabels(df.columns)

            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    try:
                        value = str(df.iloc[row, col])
                        if value in ['NaT', 'nan', 'None', '']:
                            value = ''
                    except Exception as e:
                        value = f"数据错误: {str(e)}"
                    self.result_table.setItem(row, col, QTableWidgetItem(value))

            self.result_table.resizeColumnsToContents()
        else:
            # 清空表格
            self.result_table.clear()
            self.result_table.setRowCount(0)
            self.result_table.setColumnCount(0)

    def show_table(self):
        """显示表格视图，隐藏图表视图"""
        self.result_table.setVisible(True)
        self.chart_widget.setVisible(False)
        self.table_btn.setChecked(True)
        self.chart_btn.setChecked(False)

    def show_chart(self):
        """显示图表视图，隐藏表格视图"""
        self.result_table.setVisible(False)
        self.chart_widget.setVisible(True)
        self.table_btn.setChecked(False)
        self.chart_btn.setChecked(True)

    def change_save_dir(self):
        """通过浏览更改当前保存目录（不影响默认目录）"""
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", self.current_save_dir  # 使用当前目录作为初始路径
        )
        if new_dir:
            self.save_dir_edit.setText(new_dir)
            self.apply_save_dir()

    def apply_save_dir(self):
        """应用当前保存目录更改（仅更新临时目录，不修改默认目录）"""
        new_dir = self.save_dir_edit.text().strip()
        if new_dir and os.path.isdir(new_dir):
            # 仅更新当前目录，不修改配置中的默认目录
            self.current_save_dir = new_dir
        else:
            show_error_message(self, "错误", "无效的目录路径")
            self.save_dir_edit.setText(self.current_save_dir)  # 恢复当前目录

    def save_results(self):
        if not self.current_result:
            show_error_message(self, "错误", "没有可保存的结果")
            return

        try:
            # 检查当前视图模式
            if self.table_btn.isChecked() and "result_table" in self.current_result and self.current_result[
                "result_table"] is not None:
                # 表格视图 - 保存表格
                df = self.current_result["result_table"]
                base_name = "analysis_result"
                filename = get_unique_filename(self.current_save_dir, base_name, "csv")
                file_path = os.path.join(self.current_save_dir, filename)
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
                show_info_message(self, "成功", f"表格已保存至:\n{file_path}")

            elif self.chart_btn.isChecked() and self.chart_widget.current_chart is not None:
                # 图表视图 - 保存图表
                base_name = "analysis_chart"
                filename = get_unique_filename(self.current_save_dir, base_name, "png")
                file_path = os.path.join(self.current_save_dir, filename)
                self.chart_widget.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                show_info_message(self, "成功", f"图表已保存至:\n{file_path}")

            else:
                show_error_message(self, "错误", "没有可保存的有效内容")

        except Exception as e:
            show_error_message(self, "保存失败", f"无法保存结果: {str(e)}")

    def start_new_analysis(self):
        """返回分析要求页并清除当前分析结果（保留已选文件）"""
        # 仅清除当前分析结果，不影响已选择的文件
        self.current_result = None
        self.summary_display.clear()
        self.result_table.clear()
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(0)
        self.save_btn.setEnabled(False)
        self.chart_btn.setEnabled(False)
        self.show_table()

        # 切换到分析要求页（数据分析标签页，索引为2）
        if self.parent and hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(2)