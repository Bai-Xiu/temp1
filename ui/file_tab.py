from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QListWidget, QGroupBox, QSplitter,
                             QFileDialog, QListWidgetItem, QMessageBox, )
from PyQt5.QtCore import (Qt, QTimer)
from PyQt5.QtWidgets import QFileIconProvider
import os
import shutil
from utils.helpers import get_file_list, show_info_message, show_error_message
from ui.sensitive_tab import ProgressDialog  # 导入新类
from PyQt5.QtCore import QThread, pyqtSignal


class FileTab(QWidget):
    def __init__(self, processor, config, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.config = config  # 配置对象（存储默认目录）
        self.selected_files = []
        self.parent = parent  # 保存父窗口引用
        self.current_data_dir = self.config.get("data_dir")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 数据目录选择区
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("当前数据目录:"))

        # 显示当前数据目录（而非直接读取配置）
        self.data_dir_edit = QLineEdit(self.current_data_dir)
        self.data_dir_edit.setReadOnly(False)
        dir_layout.addWidget(self.data_dir_edit)

        self.change_dir_btn = QPushButton("浏览...")
        self.change_dir_btn.clicked.connect(self.change_data_dir)
        dir_layout.addWidget(self.change_dir_btn)

        self.apply_dir_btn = QPushButton("应用")
        self.apply_dir_btn.clicked.connect(self.apply_data_dir)
        dir_layout.addWidget(self.apply_dir_btn)

        # 添加文件按钮
        add_file_layout = QHBoxLayout()
        self.add_external_btn = QPushButton("添加外部文件...")
        self.add_external_btn.clicked.connect(self.add_external_files)
        add_file_layout.addWidget(self.add_external_btn)
        add_file_layout.addStretch()

        # 顶部按钮区
        btn_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新文件列表")
        self.refresh_btn.clicked.connect(self.update_file_list)
        self.add_btn = QPushButton("添加选中")
        self.add_btn.clicked.connect(self.add_files)
        self.remove_btn = QPushButton("移除选中")
        self.remove_btn.clicked.connect(self.remove_files)
        self.clear_btn = QPushButton("清空选择")
        self.clear_btn.clicked.connect(self.clear_selection)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)

        # 文件列表区域
        list_group = QGroupBox("可用日志文件")
        list_layout = QVBoxLayout(list_group)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        list_layout.addWidget(self.file_list)

        # 已选文件区域
        selected_group = QGroupBox("已选择文件")
        selected_layout = QVBoxLayout(selected_group)

        self.selected_list = QListWidget()
        selected_layout.addWidget(self.selected_list)

        # 分割器布局
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(list_group)
        splitter.addWidget(selected_group)
        splitter.setSizes([300, 200])

        # 底部按钮
        self.next_btn = QPushButton("下一步分析")
        self.next_btn.clicked.connect(self.go_to_analysis)
        self.next_btn.setEnabled(False)

        # 组装布局
        layout.addLayout(dir_layout)
        layout.addLayout(add_file_layout)
        layout.addLayout(btn_layout)
        layout.addWidget(splitter)
        layout.addWidget(self.next_btn)

        # 在按钮布局添加去敏相关按钮
        self.anonymize_btn = QPushButton("去敏选中文件")
        self.anonymize_btn.clicked.connect(self.anonymize_selected_files)
        self.anonymize_btn.setEnabled(False)
        btn_layout.addWidget(self.anonymize_btn)

        # 初始加载文件列表
        self.update_file_list()

    def change_data_dir(self):
        """通过浏览更改当前数据目录（不影响默认目录）"""
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择数据目录", self.current_data_dir  # 使用当前目录作为初始路径
        )
        if new_dir:
            self.data_dir_edit.setText(new_dir)
            self.apply_data_dir()


    def apply_data_dir(self):
        """应用当前数据目录更改（仅更新临时目录，不修改默认目录）"""
        new_dir = self.data_dir_edit.text().strip()
        if new_dir and os.path.isdir(new_dir):
            # 仅更新当前目录，不修改配置中的默认目录
            self.current_data_dir = new_dir
            # 更新处理器的当前工作目录
            self.processor.set_current_data_dir(new_dir)
            self.update_file_list()
            self.clear_selection()
            if self.parent:
                self.parent.statusBar().showMessage(f"当前数据目录已更新为: {new_dir}")
        else:
            show_error_message(self, "警告", "无效的目录路径")
            self.data_dir_edit.setText(self.current_data_dir)  # 恢复当前目录

    def add_external_files(self):
        """添加外部文件到当前数据目录（而非默认目录）"""
        supported_exts = [
            "CSV文件 (*.csv)",
            "Excel文件 (*.xlsx *.xls)",
            "JSON文件 (*.json)",
            "文本日志 (*.txt *.log)",
            "所有支持的文件 (*.csv *.xlsx *.xls *.json *.txt *.log)",
            "所有文件 (*)"
        ]
        file_filter = ";;".join(supported_exts)

        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择日志文件", "", file_filter
        )

        if file_paths:
            # 使用当前数据目录作为目标路径
            data_dir = self.current_data_dir
            added_files = []

            for file_path in file_paths:
                try:
                    # 获取文件名
                    file_name = os.path.basename(file_path)
                    dest_path = os.path.join(data_dir, file_name)

                    # 检查文件是否已存在
                    if os.path.exists(dest_path):
                        # 询问是否覆盖
                        reply = QMessageBox.question(
                            self, "文件已存在",
                            f"文件 {file_name} 已存在，是否覆盖？",
                            QMessageBox.Yes | QMessageBox.No
                        )
                        if reply != QMessageBox.Yes:
                            continue

                    # 复制文件
                    shutil.copy2(file_path, dest_path)
                    added_files.append(file_name)
                except Exception as e:
                    show_error_message(self, "复制失败", f"无法复制文件 {file_path}: {str(e)}")

            if added_files:
                show_info_message(
                    self, "添加成功",
                    f"已添加 {len(added_files)} 个文件到数据目录"
                )
                self.update_file_list()

    def update_file_list(self):
        """更新当前数据目录中的文件列表"""
        self.file_list.clear()
        try:
            # 读取当前数据目录中的文件
            files = get_file_list(self.current_data_dir)
            icon_provider = QFileIconProvider()

            for file in files:
                item = QListWidgetItem(icon_provider.icon(QFileIconProvider.File), file)
                self.file_list.addItem(item)

            if self.parent:
                self.parent.statusBar().showMessage(f"已加载 {len(files)} 个文件")
        except Exception as e:
            show_error_message(self, "警告", f"加载文件列表失败: {str(e)}")

    def add_files(self):
        """添加文件到选择列表"""
        selected = self.file_list.selectedItems()
        if not selected:
            show_info_message(self, "提示", "请先选择文件")
            return

        for item in selected:
            filename = item.text()
            if not self.selected_list.findItems(filename, Qt.MatchExactly):
                self.selected_list.addItem(filename)
                self.selected_files.append(filename)

        self.update_next_button()

    def remove_files(self):
        """移除选中的文件"""
        for item in self.selected_list.selectedItems():
            self.selected_files.remove(item.text())
            self.selected_list.takeItem(self.selected_list.row(item))

        self.update_next_button()

    def clear_selection(self):
        """清空选择列表"""
        self.selected_list.clear()
        self.selected_files = []
        self.update_next_button()

    def update_next_button(self):
        """更新下一步按钮状态"""
        self.next_btn.setEnabled(len(self.selected_files) > 0)
        has_files = len(self.selected_files) > 0  # 补充缺失的 has_files 定义
        self.next_btn.setEnabled(has_files)
        self.anonymize_btn.setEnabled(has_files)

    def go_to_analysis(self):
        """前往分析标签页"""
        if self.parent and hasattr(self.parent, 'tabs'):
            self.parent.tabs.setCurrentIndex(2)  # 假设分析标签页索引为2
            if hasattr(self.parent, 'statusBar'):
                self.parent.statusBar().showMessage(f"已选择 {len(self.selected_files)} 个文件")

    def get_selected_files(self):
        """获取选中的文件列表"""
        return self.selected_files.copy()

    def anonymize_selected_files(self):
        """对选中的文件进行去敏处理"""
        if not self.selected_files:
            show_info_message(self, "提示", "请先选择文件")
            return

        # 选择保存目录
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择去敏文件保存目录",
            self.config.get("save_dir", "")
        )

        if not save_dir:
            return

        # 创建进度对话框
        progress_dialog = ProgressDialog("正在进行去敏处理", len(self.selected_files), self)
        progress_dialog.show()

        # 创建并启动去敏线程
        self.anonymize_thread = AnonymizeThread(
            self.processor,
            self.selected_files,
            save_dir
        )
        self.anonymize_thread.update_signal.connect(progress_dialog.update_progress)
        self.anonymize_thread.complete_signal.connect(lambda res: self.on_anonymize_complete(res, progress_dialog))
        self.anonymize_thread.start()

    # 确保这个方法在FileTab类内部
    def on_anonymize_complete(self, result, progress_dialog):
        """去敏完成后的处理"""
        progress_dialog.complete()

        if result["status"] == "success":
            show_info_message(
                self, "成功",
                f"已完成{len(result['results'])}个文件的去敏处理"
            )
            # 更新配置中的保存目录
            if result['results']:
                first_path = next(iter(result['results'].values()))
                self.config.set("save_dir", os.path.dirname(first_path))
        else:
            show_error_message(self, "错误", result["message"])

        # 延迟关闭进度对话框
        QTimer.singleShot(1000, progress_dialog.close)

# 添加去敏处理线程
class AnonymizeThread(QThread):
    update_signal = pyqtSignal(str)
    complete_signal = pyqtSignal(dict)

    def __init__(self, processor, file_names, output_dir):
        super().__init__()
        self.processor = processor
        self.file_names = file_names
        self.output_dir = output_dir

    def run(self):
        try:
            results = {}
            for filename in self.file_names:
                self.update_signal.emit(filename)
                # 单文件处理逻辑（从process_and_anonymize_files拆分）
                data_dict = self.processor._load_file_data([filename])
                for fname, df in data_dict.items():
                    anonymized_df = self.processor._anonymize_dataframe(df)
                    base_name = os.path.splitext(fname)[0]
                    ext = os.path.splitext(fname)[1]
                    output_path = os.path.join(
                        self.output_dir,
                        f"{base_name}_anonymized{ext}"
                    )

                    # 保存文件
                    if ext.lower() in ['.csv']:
                        anonymized_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    elif ext.lower() in ['.xlsx', '.xls']:
                        anonymized_df.to_excel(output_path, index=False)
                    elif ext.lower() in ['.json']:
                        anonymized_df.to_json(output_path, orient='records', force_ascii=False)
                    else:
                        content = "\n".join(anonymized_df.iloc[:, 0].astype(str).tolist())
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    results[fname] = output_path
            self.complete_signal.emit({"status": "success", "results": results})
        except Exception as e:
            self.complete_signal.emit({"status": "error", "message": str(e)})
