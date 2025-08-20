# 修改 log_ai_system/utils/config.py 文件
import os
import json

class Config:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.json')
        # 初始化为空目录，不自动设置
        self.config = {
            "api_key": "",
            "data_dir": "",
            "save_dir": "",
            "verbose_logging": False
        }
        self.load()
        if self.config["data_dir"]:
            os.makedirs(self.config["data_dir"], exist_ok=True)
        if self.config["save_dir"]:
            os.makedirs(self.config["save_dir"], exist_ok=True)

    def load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except Exception as e:
                print(f"加载配置失败: {str(e)}")

    def save(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {str(e)}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save()  # 自动保存