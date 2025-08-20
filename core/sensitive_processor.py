import re
import json
import os
import random
import string
import pandas as pd
import uuid
from datetime import datetime
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

        self.compiled_patterns = {}  # 存储预编译的正则表达式
        self._compile_patterns()

        # 确保文件存在并加载敏感词
        self._ensure_file_exists()
        self.load_sensitive_words()

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

    def _generate_replacement(self):
        """生成随机替换词: PROTECTED{8位随机大小写字母+数字}"""
        chars = string.ascii_letters + string.digits
        random_str = ''.join(random.choices(chars, k=8))
        return f"PROTECTED{random_str}"

    def _sort_sensitive_words(self):
        """按敏感词长度降序排序，避免子串冲突"""
        sorted_words = sorted(
            self.sensitive_words.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        self.sensitive_words = dict(sorted_words)
        # 更新替换映射
        self.replacement_map = {v: k for k, v in self.sensitive_words.items()}

    def load_sensitive_words(self):
        """从文件加载敏感词"""
        try:
            with open(self.sensitive_file, 'r', encoding='utf-8') as f:
                self.sensitive_words = json.load(f)

            # 去重并排序
            self.sensitive_words = {k: v for k, v in self.sensitive_words.items()}
            self._sort_sensitive_words()
            self._compile_patterns()  # 加载后重新编译
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
        """从CSV/Excel导入敏感词"""
        if not os.path.exists(file_path):
            return False, "文件不存在"

        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            # 尝试不同编码
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

            count = 0
            for _, row in df.iterrows():
                word = str(row["敏感词"]).strip()
                replacement = str(row.get("替换词", "")).strip()

                if not word:
                    continue

                # 如果已存在则跳过
                if word in self.sensitive_words:
                    continue

                # 添加敏感词
                self.add_sensitive_word(word, replacement)
                count += 1

            self._sort_sensitive_words()
            self.save_sensitive_words()
            self._compile_patterns()
            return True, f"成功导入 {count} 个敏感词"
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
        """替换文本中的敏感词"""
        if not text or not isinstance(text, str) or not self.sensitive_words:
            return text, {}

        replaced_text = text
        replace_count = {}

        # 使用预编译的正则表达式
        for word, (pattern, replacement) in self.compiled_patterns.items():
            try:
                # 执行替换并计数
                temp_text, count = pattern.subn(replacement, replaced_text)

                if count > 0:
                    replace_count[word] = replace_count.get(word, 0) + count
                    replaced_text = temp_text

            except Exception as e:
                print(f"替换敏感词'{word}'时出错: {str(e)}")

        return replaced_text, replace_count

    def restore_sensitive_words(self, text):
        """将文本中的替换词还原为原始敏感词，优化匹配精度避免漏匹配"""
        if not text or not isinstance(text, str) or not self.replacement_map:
            return text

        restored_text = text
        replace_count = {}

        # 1. 构建替换词的合并正则模式（长替换词优先，确保完整匹配）
        sorted_replacements = sorted(
            self.replacement_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        if sorted_replacements:
            # 为每个替换词添加单词边界确保精确匹配，避免部分包含
            escaped_replacements = [f'\\b{re.escape(rep)}\\b' for rep, _ in sorted_replacements]
            # 移除IGNORECASE避免大小写不匹配导致的KeyError，移除不必要的MULTILINE
            combined_pattern = re.compile('|'.join(escaped_replacements))

            # 2. 单次遍历还原，增加异常处理
            def restore_callback(match):
                matched_rep = match.group()
                try:
                    original_word = self.replacement_map[matched_rep]
                    replace_count[matched_rep] = replace_count.get(matched_rep, 0) + 1
                    return original_word
                except KeyError:
                    # 记录无法匹配的替换词，便于排查问题
                    print(f"警告：未找到替换词'{matched_rep}'对应的原始敏感词")
                    return matched_rep  # 无法还原时保持原样

            restored_text = combined_pattern.sub(restore_callback, text)

        return restored_text

    def get_all_sensitive_words(self):
        """获取所有敏感词列表"""
        return [(k, v) for k, v in self.sensitive_words.items()]

