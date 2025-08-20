from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTableWidget, QTableWidgetItem, QFileDialog,
                             QGroupBox, QMessageBox, QHeaderView, QGridLayout, QApplication, QMenu)
from PyQt5.QtCore import Qt
import os
from utils.helpers import show_info_message, show_error_message
from PyQt5.QtWidgets import QDialog, QProgressBar, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


class SensitiveWordTab(QWidget):
    def __init__(self, sensitive_processor, parent=None):
        super().__init__(parent)
        self.sensitive_processor = sensitive_processor
        self.parent = parent
        self.init_ui()
        self.refresh_table()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 操作按钮区
        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("添加敏感词")
        self.add_btn.clicked.connect(self.add_word_dialog)

        self.import_btn = QPushButton("导入敏感词")
        self.import_btn.clicked.connect(self.import_words)

        self.export_btn = QPushButton("导出敏感词")
        self.export_btn.clicked.connect(self.export_words)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()

        # 表格区域
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["敏感词", "替换词"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        # 添加到主布局
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.table)

    def refresh_table(self):
        """刷新表格数据"""
        words = self.sensitive_processor.get_all_sensitive_words()
        self.table.setRowCount(len(words))

        for row, (word, replacement) in enumerate(words):
            word_item = QTableWidgetItem(word)
            replacement_item = QTableWidgetItem(replacement)

            word_item.setFlags(word_item.flags() & ~Qt.ItemIsEditable)
            replacement_item.setFlags(replacement_item.flags() & ~Qt.ItemIsEditable)

            self.table.setItem(row, 0, word_item)
            self.table.setItem(row, 1, replacement_item)

    def show_context_menu(self, position):
        """显示右键菜单"""
        item = self.table.itemAt(position)
        if not item:
            return

        row = item.row()
        word_item = self.table.item(row, 0)
        if not word_item:
            return

        word = word_item.text()

        # 创建菜单
        menu = QMenu(self)
        edit_action = menu.addAction("修改")
        copy_action = menu.addAction("复制替换词")
        delete_action = menu.addAction("删除")

        action = menu.exec_(self.table.mapToGlobal(position))

        if action == edit_action:
            self.edit_word_dialog(word)
        elif action == copy_action:
            replacement = self.table.item(row, 1).text()
            clipboard = QApplication.clipboard()
            clipboard.setText(replacement)
            show_info_message(self, "成功", "替换词已复制到剪贴板")
        elif action == delete_action:
            self.delete_word(word)

    def add_word_dialog(self):
        """添加敏感词对话框"""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("添加敏感词")

        # 创建输入框
        word_input = QLineEdit()
        replacement_input = QLineEdit()
        word_input.setPlaceholderText("请输入敏感词")
        replacement_input.setPlaceholderText("可选")

        # 设置布局
        layout = QGridLayout()
        layout.addWidget(QLabel("敏感词:"), 0, 0)
        layout.addWidget(word_input, 0, 1)
        layout.addWidget(QLabel("替换词:"), 1, 0)
        layout.addWidget(replacement_input, 1, 1)

        dialog.layout().addLayout(layout, 1, 0, 1, 2)
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if dialog.exec_() == QMessageBox.Ok:
            word = word_input.text().strip()
            replacement = replacement_input.text().strip()

            success, msg = self.sensitive_processor.add_sensitive_word(word, replacement)
            if success:
                self.refresh_table()
            show_info_message(self, "结果", msg)

    def edit_word_dialog(self, old_word):
        """编辑敏感词对话框"""
        # 获取当前替换词
        replacement = None
        for word, rep in self.sensitive_processor.get_all_sensitive_words():
            if word == old_word:
                replacement = rep
                break

        if not replacement:
            show_error_message(self, "错误", "未找到该敏感词")
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("修改敏感词")

        # 创建输入框
        word_input = QLineEdit(old_word)
        replacement_input = QLineEdit(replacement)

        # 设置布局
        layout = QGridLayout()
        layout.addWidget(QLabel("敏感词:"), 0, 0)
        layout.addWidget(word_input, 0, 1)
        layout.addWidget(QLabel("替换词:"), 1, 0)
        layout.addWidget(replacement_input, 1, 1)

        dialog.layout().addLayout(layout, 1, 0, 1, 2)
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        if dialog.exec_() == QMessageBox.Ok:
            new_word = word_input.text().strip()
            new_replacement = replacement_input.text().strip()

            success, msg = self.sensitive_processor.update_sensitive_word(
                old_word, new_word, new_replacement
            )
            if success:
                self.refresh_table()
            show_info_message(self, "结果", msg)

    def delete_word(self, word):
        """删除敏感词"""
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除敏感词 '{word}' 吗？",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 调用处理器的删除方法
            success, msg = self.sensitive_processor.remove_sensitive_word(word)
            if success:
                self.refresh_table()
                show_info_message(self, "成功", f"敏感词 '{word}' 已删除")
            else:
                show_error_message(self, "失败", msg)

    def import_words(self):
        """导入敏感词"""
        supported_exts = [
            "CSV文件 (*.csv)",
            "Excel文件 (*.xlsx *.xls)",
            "所有支持的文件 (*.csv *.xlsx *.xls)"
        ]
        file_filter = ";;".join(supported_exts)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择导入文件", "", file_filter
        )

        if file_path:
            success, msg = self.sensitive_processor.import_from_file(file_path)
            if success:
                self.refresh_table()
            show_info_message(self, "导入结果", msg)

    def export_words(self):
        """导出敏感词"""
        if not self.sensitive_processor.get_all_sensitive_words():
            show_info_message(self, "提示", "没有敏感词可导出")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存导出文件", "sensitive_words.csv",
            "CSV文件 (*.csv);;Excel文件 (*.xlsx)"
        )

        if file_path:
            success, msg = self.sensitive_processor.export_to_file(file_path)
            show_info_message(self, "导出结果", msg)


class ProgressDialog(QDialog):
    def __init__(self, title, total_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 100)
        self.total = total_files
        self.current = 0

        layout = QVBoxLayout()

        self.label = QLabel("准备处理...")
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, total_files)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

    def update_progress(self, filename):
        self.current += 1
        self.label.setText(f"正在处理: {filename}")
        self.progress_bar.setValue(self.current)
        # 处理UI事件，防止界面冻结
        QApplication.processEvents()

    def complete(self):
        self.label.setText("处理完成!")
        self.progress_bar.setValue(self.total)