import pandas as pd
import json
from abc import ABC, abstractmethod


class FileProcessor(ABC):
    """文件处理器基类，所有文件类型处理器需继承此类"""

    @abstractmethod
    def get_supported_extensions(self):
        """返回支持的文件扩展名列表（如 ['.csv']）"""
        pass

    @abstractmethod
    def read_file(self, file_path, encodings=None, **kwargs):
        """读取文件并返回DataFrame
        Args:
            file_path: 文件路径
            encodings: 尝试的编码列表
            kwargs: 额外参数
        Returns:
            pd.DataFrame: 读取的数据
        Raises:
            Exception: 读取失败时抛出
        """
        pass


class CsvFileProcessor(FileProcessor):
    def get_supported_extensions(self):
        return ['.csv']

    def read_file(self, file_path, encodings=None, **kwargs):
        # 扩展编码列表，增加utf-16等Windows可能使用的编码
        encodings = encodings or ['utf-8', 'gbk', 'gb2312', 'ansi', 'utf-16', 'utf-16-le']
        # 提供默认分隔符参数，允许用户通过kwargs覆盖
        sep = kwargs.get('sep', ',')
        # 允许灵活设置表头（默认自动识别，失败则强制无表头）
        header = kwargs.get('header', 'infer')

        for encoding in encodings:
            try:
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=sep,
                    engine=kwargs.get('engine', 'python'),
                    header=header,
                    # 忽略空行，增强容错性
                    skip_blank_lines=True
                )
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                # 细化异常捕获，避免非编码问题被忽略
                continue
        raise ValueError(f"CSV文件读取失败，已尝试编码: {encodings}")


class ExcelFileProcessor(FileProcessor):
    def get_supported_extensions(self):
        return ['.xlsx', '.xls']

    def read_file(self, file_path, **kwargs):
        # 支持更多引擎（xlrd用于旧版xls，openpyxl用于xlsx）
        engines = kwargs.get('engines', ['openpyxl', 'xlrd'])
        sheet_name = kwargs.get('sheet_name', 0)

        for engine in engines:
            try:
                return pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    engine=engine,
                    # 忽略空行
                    skiprows=lambda x: x in kwargs.get('skip_rows', []),
                    keep_default_na=False  # 避免将空字符串识别为NaN
                )
            except (ValueError, ImportError, pd.errors.ParserError) as e:
                continue
        raise ValueError(f"Excel文件读取失败，已尝试引擎: {engines}")

class JsonFileProcessor(FileProcessor):
    def get_supported_extensions(self):
        return ['.json']

    def read_file(self, file_path, encodings=None, **kwargs):
        encodings = encodings or ['utf-8', 'gbk', 'gb2312', 'ansi', 'utf-16']
        # 允许设置JSON解析的严格模式（默认非严格，兼容更多格式）
        strict = kwargs.get('strict', False)

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    # 非严格模式解析，容忍尾逗号等常见问题
                    data = json.load(f, strict=strict)
                # 支持更多JSON结构（如嵌套字典）
                if isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, dict):
                    # 嵌套字典转为多列
                    return pd.json_normalize(data)
                else:
                    raise ValueError("JSON格式不支持（需为列表或对象）")
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        raise ValueError(f"JSON文件读取失败，已尝试编码: {encodings}")

class TxtFileProcessor(FileProcessor):
    def get_supported_extensions(self):
        return ['.txt', '.log']

    def read_file(self, file_path, encodings=None, **kwargs):
        encodings = encodings or ['utf-8', 'gbk', 'gb2312', 'ansi']
        delimiter = kwargs.get('delimiter', '\t')
        for encoding in encodings:
            try:
                return pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=delimiter,
                    engine='python',
                    header=None,
                    names=['event']
                )
            except Exception:
                continue
        raise ValueError(f"TXT/LOG文件读取失败，已尝试编码: {encodings}")