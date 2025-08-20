from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
                             QComboBox, QProgressBar, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt
from core.analysis_thread import AnalysisThread
from utils.helpers import show_error_message


class AnalysisTab(QWidget):
    def __init__(self, processor, file_tab, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.file_tab = file_tab  # 引用文件选择标签页
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """初始化分析标签页UI"""
        layout = QVBoxLayout(self)

        # 分析请求输入
        req_group = QGroupBox("分析请求")
        req_layout = QVBoxLayout(req_group)

        self.request_input = QTextEdit()
        self.request_input.setPlaceholderText("请输入分析需求，例如：统计各类型入侵次数、列出出现频率最高的10个IP地址...")
        req_layout.addWidget(self.request_input)

        # 处理模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["代码处理", "直接回答"])
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setVisible(False)

        # 按钮区
        btn_layout = QHBoxLayout()

        self.back_btn = QPushButton("返回")
        self.back_btn.clicked.connect(self.go_back)

        self.start_btn = QPushButton("开始分析")
        self.start_btn.clicked.connect(self.start_analysis)

        btn_layout.addWidget(self.back_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)

        # 组装布局
        layout.addWidget(req_group)
        layout.addLayout(mode_layout)
        layout.addWidget(self.progress)
        layout.addLayout(btn_layout)

    def go_back(self):
        """返回文件选择标签页"""
        if self.parent and hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(1)  # 假设文件选择标签页索引为1

    def start_analysis(self):
        """开始数据分析"""
        request = self.request_input.toPlainText().strip()
        if not request:
            show_error_message(self, "警告", "请输入分析请求")
            return

        selected_files = self.file_tab.get_selected_files()
        if not selected_files:
            show_error_message(self, "警告", "请先选择文件")
            return

        # 准备分析
        self.start_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # 无限进度
        if self.parent and hasattr(self.parent, 'statusBar'):
            self.parent.statusBar().showMessage("分析中...")

        # 确定模式
        mode = "1" if self.mode_combo.currentText().startswith("代码处理") else "2"

        # 启动后台线程
        self.analysis_thread = AnalysisThread(
            self.processor,
            selected_files,
            request,
            mode
        )
        self.analysis_thread.update_signal.connect(self.update_status)
        self.analysis_thread.complete_signal.connect(self.analysis_complete)
        self.analysis_thread.start()

    def update_status(self, message):
        """更新状态信息"""
        if self.parent and hasattr(self.parent, 'statusBar'):
            self.parent.statusBar().showMessage(message)

    def on_mode_changed(self, index):
        """模式切换时更新图表按钮状态"""
        is_direct_mode = (index == 1)  # 直接回答模式索引为1
        if self.parent and hasattr(self.parent, 'results_tab'):
            # 禁用图表按钮
            self.parent.results_tab.chart_btn.setEnabled(not is_direct_mode)
            # 强制显示表格视图
            self.parent.results_tab.show_table()

    def analysis_complete(self, result):
        """分析完成处理"""
        self.progress.setVisible(False)
        self.start_btn.setEnabled(True)

        if result["status"] == "success":
            if self.parent:
                self.parent.statusBar().showMessage("分析完成")
                self.parent.set_analysis_result(result["result"])
                self.parent.tabs.setCurrentIndex(3)  # 切换到结果标签页
        else:
            if self.parent and hasattr(self.parent, 'statusBar'):
                self.parent.statusBar().showMessage("分析失败")
            show_error_message(self, "错误", result["message"])