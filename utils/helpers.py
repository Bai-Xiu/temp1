import os
import re
from PyQt5.QtWidgets import QMessageBox


def show_error_message(parent, title, message):
    """显示错误消息框"""
    QMessageBox.critical(parent, title, message)


def show_info_message(parent, title, message):
    """显示信息消息框"""
    QMessageBox.information(parent, title, message)


def get_file_list(directory):
    """获取目录中的文件列表"""
    if not os.path.exists(directory):
        return []

    try:
        # 只返回文件，不返回目录
        return [f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and not f.startswith('.')]
    except Exception as e:
        print(f"获取文件列表失败: {str(e)}")
        return []


def sanitize_filename(filename):
    """清理文件名"""
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    return safe_name.strip() or "unnamed_file"


def get_unique_filename(directory, base_name, extension):
    """生成唯一文件名"""
    if extension.startswith('.'):
        extension = extension[1:]

    counter = 1
    filename = f"{base_name}.{extension}"
    file_path = os.path.join(directory, filename)

    while os.path.exists(file_path):
        filename = f"{base_name}_{counter}.{extension}"
        file_path = os.path.join(directory, filename)
        counter += 1

    return filename


def is_valid_file(file_path):
    """验证支持的文件类型"""
    if not os.path.exists(file_path):
        return False, "文件不存在"

    if not os.path.isfile(file_path):
        return False, "路径指向的不是文件"

    # 检查文件大小
    if os.path.getsize(file_path) == 0:
        return False, "文件为空"

    # 检查扩展名（可扩展）
    supported_exts = {'.csv', '.xlsx', '.xls', '.json', '.txt', '.log'}
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in supported_exts:
        return False, f"不支持的文件格式: {ext}。支持: {', '.join(supported_exts)}"

    return True, "有效的文件"