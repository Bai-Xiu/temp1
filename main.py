import sys
import os
from PyQt5.QtWidgets import QApplication
from ui.main_window import LogAnalyzerGUI
from utils.config import Config

def main():
    config = Config()
    config.load()

    app = QApplication(sys.argv)
    window = LogAnalyzerGUI(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()