from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar)
from PyQt5.QtGui import QFont
from PyQt5.QtGui import QIcon
import os
from ui.config_tab import ConfigTab  # 新增配置标签页类
from ui.file_tab import FileTab
from ui.analysis_tab import AnalysisTab
from ui.results_tab import ResultsTab
from core.processor import LogAIProcessor
from ui.sensitive_tab import SensitiveWordTab
class LogAnalyzerGUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = LogAIProcessor(config)
        self.init_ui()
        self.set_window_icon()


    def init_ui(self):
        # 窗口基本设置
        self.setWindowTitle("信息安全日志AI分析系统")
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(QFont("SimHei", 9))
        self.set_window_icon()

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标签页容器
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # 初始化标签页
        self.init_tabs()

        # 状态栏
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

    def init_tabs(self):
        """初始化所有标签页"""
        # 配置标签页
        self.config_tab = ConfigTab(self.config, self)
        self.tabs.addTab(self.config_tab, "配置")

        # 文件选择标签页
        self.file_tab = FileTab(self.processor, self.config, self)
        self.tabs.addTab(self.file_tab, "文件选择")

        # 分析标签页
        self.analysis_tab = AnalysisTab(self.processor, self.file_tab, self)
        self.tabs.addTab(self.analysis_tab, "数据分析")

        # 结果标签页
        self.results_tab = ResultsTab(self.config, self)
        self.tabs.addTab(self.results_tab, "分析结果")

        # 添加敏感词管理标签页
        self.sensitive_tab = SensitiveWordTab(self.processor.sensitive_processor, self)
        self.tabs.addTab(self.sensitive_tab, "敏感词管理")

    def set_window_icon(self):
        # 解析图标绝对路径（项目根目录 -> resources -> app_icon.ico）
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取项目根目录
        icon_path = os.path.join(root_dir, "resources", "app_icon.ico")

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"警告：图标文件不存在 - {icon_path}")

    def set_analysis_result(self, result):
        """将分析结果传递给结果标签页"""
        self.results_tab.set_result(result)