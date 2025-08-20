from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                            QPushButton, QGroupBox, QFileDialog)
from utils.helpers import show_info_message, show_error_message
import os
from core.api_client import DeepSeekAPI

class ConfigTab(QWidget):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.parent = parent
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)

        # API Key设置
        api_group = QGroupBox("DeepSeek API 设置")
        api_layout = QVBoxLayout(api_group)

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))

        self.api_key_edit = QLineEdit(self.config.get("api_key"))
        self.api_key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        api_key_layout.addWidget(self.api_key_edit)

        self.save_api_btn = QPushButton("保存")
        self.save_api_btn.clicked.connect(self.save_api_key)
        api_key_layout.addWidget(self.save_api_btn)

        api_layout.addLayout(api_key_layout)

        # 其他配置选项
        other_group = QGroupBox("其他配置")
        other_layout = QVBoxLayout(other_group)

        # 数据目录配置
        data_dir_layout = QHBoxLayout()
        data_dir_layout.addWidget(QLabel("默认数据目录:"))

        self.default_data_dir_edit = QLineEdit(self.config.get("data_dir"))
        self.default_data_dir_edit.setReadOnly(True)
        data_dir_layout.addWidget(self.default_data_dir_edit)

        self.change_default_data_dir_btn = QPushButton("更改...")
        self.change_default_data_dir_btn.clicked.connect(self.change_default_data_dir)
        data_dir_layout.addWidget(self.change_default_data_dir_btn)

        # 结果保存目录配置
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(QLabel("默认结果目录:"))

        self.default_save_dir_edit = QLineEdit(self.config.get("save_dir"))
        self.default_save_dir_edit.setReadOnly(True)
        save_dir_layout.addWidget(self.default_save_dir_edit)

        self.change_default_save_dir_btn = QPushButton("更改...")
        self.change_default_save_dir_btn.clicked.connect(self.change_default_save_dir)
        save_dir_layout.addWidget(self.change_default_save_dir_btn)

        other_layout.addLayout(data_dir_layout)
        other_layout.addLayout(save_dir_layout)

        layout.addWidget(api_group)
        layout.addWidget(other_group)
        layout.addStretch()

    def save_api_key(self):
        api_key = self.api_key_edit.text().strip()
        self.config.set("api_key", api_key)

        # 更新处理器的API Key并重新初始化客户端，传入敏感词处理器
        if hasattr(self.parent, 'processor'):
            self.parent.processor.api_key = api_key
            self.parent.processor.client = DeepSeekAPI(
                api_key=api_key,
                sensitive_processor=self.parent.processor.sensitive_processor
            ) if api_key else None

        show_info_message(self, "成功", "API Key已保存并生效")

    def change_default_data_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择数据目录", self.config.get("data_dir")
        )
        if new_dir:
            # 更新配置中的默认目录
            self.config.set("data_dir", new_dir)
            self.default_data_dir_edit.setText(new_dir)

            # 同步更新文件选择标签页的当前目录
            if hasattr(self.parent, 'file_tab'):
                self.parent.file_tab.current_data_dir = new_dir
                self.parent.file_tab.data_dir_edit.setText(new_dir)
                self.parent.file_tab.apply_data_dir()  # 触发刷新

            # 更新处理器的默认目录
            if hasattr(self.parent, 'processor'):
                self.parent.processor.set_default_data_dir(new_dir)
            show_info_message(self, "成功", "默认数据目录已更新")

    def change_default_save_dir(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择结果目录", self.config.get("save_dir")
        )
        if new_dir:
            # 更新配置中的默认目录
            self.config.set("save_dir", new_dir)
            self.default_save_dir_edit.setText(new_dir)

            # 同步更新结果标签页的当前目录
            if hasattr(self.parent, 'results_tab'):
                self.parent.results_tab.current_save_dir = new_dir
                self.parent.results_tab.save_dir_edit.setText(new_dir)

            # 更新处理器的默认目录
            if hasattr(self.parent, 'processor'):
                self.parent.processor.set_default_save_dir(new_dir)
            show_info_message(self, "成功", "默认结果目录已更新")
