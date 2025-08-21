import re
import json
import os
import random
import string
import pandas as pd
import uuid
from datetime import datetime
from collections import defaultdict
from utils.helpers import show_info_message, show_error_message


class SensitiveWordProcessor:
    def __init__(self, config):
        self.config = config
        self.sensitive_words = {}  # 格式: {敏感词: 替换词}
        self.replacement_map = {}  # 格式: {替换词: 敏感词} 用于还原
        self.sensitive_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../sensitive_words.json'
        )
        self.supported_encodings = ['utf-8', 'gbk', 'gb2312']

        # 确保 compiled_patterns 正确初始化（修复属性缺失错误）
        self.compiled_patterns = {}  # 存储预编译的正则表达式
        self._compile_patterns()

        # 确保文件存在并加载敏感词
        self._ensure_file_exists()
        self.load_sensitive_words()

    def _compile_combined_patterns(self):
        """编译合并的正则表达式（替换和还原用），敏感词变动时调用"""
        # 1. 编译替换用的合并正则（按长度降序，避免子串冲突）
        if self.sensitive_words:
            sorted_words = sorted(
                self.sensitive_words.keys(),
                key=lambda x: len(x),
                reverse=True
            )
            escaped_words = [re.escape(word) for word in sorted_words]
            self.combined_replace_pattern = re.compile('|'.join(escaped_words))
        else:
            self.combined_replace_pattern = None

        # 2. 编译还原用的合并正则（按长度降序）
        if self.replacement_map:
            sorted_replacements = sorted(
                self.replacement_map.keys(),
                key=lambda x: len(x),
                reverse=True
            )
            escaped_replacements = [re.escape(rep) for rep in sorted_replacements]
            # 移除\b（替换词格式固定，无歧义），提升匹配速度
            self.combined_restore_pattern = re.compile('|'.join(escaped_replacements))
        else:
            self.combined_restore_pattern = None

    def _compile_patterns(self):
        """预编译所有敏感词的正则表达式，提高替换效率"""
        self.compiled_patterns.clear()
        for word, replacement in self.sensitive_words.items():
            try:
                escaped_word = re.escape(word)
                pattern = re.compile(f'{escaped_word}')
                self.compiled_patterns[word] = (pattern, replacement)
            except Exception as e:
                print(f"编译敏感词'{word}'正则表达式出错: {str(e)}")

    def _ensure_file_exists(self):
        """确保敏感词文件存在，不存在则创建"""
        if not os.path.exists(self.sensitive_file):
            try:
                with open(self.sensitive_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"创建敏感词文件失败: {str(e)}")

    def normalize_to_replacement(self, text):
        """将文本中的原始敏感词或替换词统一转换为替换词"""
        if not text or not isinstance(text, str):
            return text

        # 先替换原始敏感词为替换词
        normalized_text, _ = self.replace_sensitive_words(text)

        # 再检查是否有残留的原始替换词（双重确保）
        for original, replacement in self.sensitive_words.items():
            if replacement in normalized_text:
                normalized_text = normalized_text.replace(replacement, self.sensitive_words[original])

        return normalized_text

    def _generate_replacement(self):
        """生成随机替换词: PROTECTED{8位随机大小写字母+数字}"""
        chars = string.ascii_letters + string.digits
        random_str = ''.join(random.choices(chars, k=8))
        return f"PROTECTED{random_str}"

    def _sort_sensitive_words(self):
        """排序后更新替换映射，并触发合并正则重编译"""
        sorted_words = sorted(
            self.sensitive_words.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self.sensitive_words = dict(sorted_words)
        self.replacement_map = {v: k for k, v in self.sensitive_words.items()}
        # 新增：重新编译合并正则
        self._compile_combined_patterns()

    def load_sensitive_words(self):
        """加载后触发合并正则编译"""
        try:
            with open(self.sensitive_file, 'r', encoding='utf-8') as f:
                self.sensitive_words = json.load(f)
            self.sensitive_words = {k: v for k, v in self.sensitive_words.items()}
            self._sort_sensitive_words()  # 会触发合并正则编译
            return True
        except Exception as e:
            return False

    def save_sensitive_words(self):
        """保存敏感词到文件"""
        try:
            with open(self.sensitive_file, 'w', encoding='utf-8') as f:
                json.dump(self.sensitive_words, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            return False

    def add_sensitive_word(self, word, replacement=None):
        """添加敏感词，自动去重和排序"""
        if not word or not isinstance(word, str) or word.strip() == "":
            return False, "敏感词不能为空"

        word = word.strip()
        if word in self.sensitive_words:
            return False, "敏感词已存在"

        # 生成替换词（如果未提供）
        if not replacement or replacement.strip() == "":
            replacement = self._generate_replacement()
        else:
            replacement = replacement.strip()

        self.sensitive_words[word] = replacement
        self._sort_sensitive_words()
        self.save_sensitive_words()
        return True, "添加成功"

    def remove_sensitive_word(self, word):
        """删除敏感词"""
        if word in self.sensitive_words:
            del self.sensitive_words[word]
            self._sort_sensitive_words()
            self.save_sensitive_words()
            return True, "删除成功"
        return False, "敏感词不存在"

    def update_sensitive_word(self, old_word, new_word, new_replacement=None):
        """更新敏感词"""
        if old_word not in self.sensitive_words:
            return False, "敏感词不存在"

        if not new_word or not isinstance(new_word, str) or new_word.strip() == "":
            return False, "新敏感词不能为空"

        new_word = new_word.strip()
        # 如果新敏感词与其他现有敏感词冲突
        if new_word != old_word and new_word in self.sensitive_words:
            return False, "新敏感词已存在"

        # 处理替换词
        if new_replacement is None:
            # 保持原替换词
            new_replacement = self.sensitive_words[old_word]
        elif new_replacement.strip() == "":
            # 生成新的替换词
            new_replacement = self._generate_replacement()
        else:
            new_replacement = new_replacement.strip()

        # 删除旧的，添加新的
        del self.sensitive_words[old_word]
        self.sensitive_words[new_word] = new_replacement
        self._sort_sensitive_words()
        self.save_sensitive_words()
        return True, "更新成功"

    def import_from_file(self, file_path):
        """从CSV/Excel导入敏感词（批量优化版）"""
        if not os.path.exists(file_path):
            return False, "文件不存在"

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            # 尝试不同编码读取文件
            df = None
            for encoding in self.supported_encodings:
                try:
                    if ext in ['.csv']:
                        df = pd.read_csv(file_path, encoding=encoding)
                    elif ext in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path)
                    break
                except Exception:
                    continue

            if df is None:
                return False, f"无法读取文件，已尝试编码: {self.supported_encodings}"

            # 检查是否包含"敏感词"列
            if "敏感词" not in df.columns:
                return False, "文件必须包含'敏感词'列"

            # 处理空值
            df = df.fillna("")

            # 批量收集新敏感词（避免逐个添加的性能损耗）
            new_words = {}
            for _, row in df.iterrows():
                word = str(row["敏感词"]).strip()
                replacement = str(row.get("替换词", "")).strip()

                if not word:
                    continue

                # 如果已存在则跳过
                if word in self.sensitive_words:
                    continue

                # 生成替换词（如果未提供）
                if not replacement:
                    replacement = self._generate_replacement()
                else:
                    replacement = replacement.strip()

                new_words[word] = replacement

            if not new_words:
                return True, "没有新的敏感词可导入"

            # 批量添加到敏感词字典
            self.sensitive_words.update(new_words)
            # 仅执行一次排序、编译和保存（核心优化点）
            self._sort_sensitive_words()
            self._compile_patterns()  # 编译正则（修复属性引用）
            self.save_sensitive_words()

            return True, f"成功导入 {len(new_words)} 个敏感词"
        except Exception as e:
            return False, f"导入失败: {str(e)}"

    def export_to_file(self, file_path):
        """导出敏感词到CSV/Excel"""
        if not self.sensitive_words:
            return False, "没有敏感词可导出"

        try:
            # 准备数据
            data = [{"敏感词": k, "替换词": v} for k, v in self.sensitive_words.items()]
            df = pd.DataFrame(data)

            # 获取文件扩展名
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            # 导出文件
            if ext == '.csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            elif ext in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False)
            else:
                return False, "不支持的文件格式，仅支持CSV和Excel"

            return True, f"成功导出 {len(self.sensitive_words)} 个敏感词"
        except Exception as e:
            return False, f"导出失败: {str(e)}"

    def replace_sensitive_words(self, text):
        """优化：单次扫描替换所有敏感词"""
        if not text or not isinstance(text, str) or not self.sensitive_words:
            return text, {}

        replace_count = defaultdict(int)

        # 回调函数：根据匹配到的敏感词返回替换词并计数
        def replace_callback(match):
            word = match.group(0)
            replacement = self.sensitive_words[word]
            replace_count[word] += 1
            return replacement

        # 使用合并正则单次替换
        replaced_text = self.combined_replace_pattern.sub(replace_callback, text)
        return replaced_text, dict(replace_count)

    def restore_sensitive_words(self, text):
        """优化：使用缓存的合并正则单次还原"""
        if not text or not isinstance(text, str) or not self.replacement_map:
            return text

        # 回调函数：根据匹配到的替换词返回原始敏感词
        def restore_callback(match):
            replacement = match.group(0)
            return self.replacement_map[replacement]

        # 使用缓存的合并正则单次还原
        return self.combined_restore_pattern.sub(restore_callback, text)

    def get_all_sensitive_words(self):
        """获取所有敏感词列表"""
        return [(k, v) for k, v in self.sensitive_words.items()]

