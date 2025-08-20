from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import matplotlib

# 设置中文字体支持
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ChartsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 配置Matplotlib支持中文显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
        self.init_ui()
        self.current_chart = None

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # 创建Matplotlib画布
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.clear_chart()

    def configure_matplotlib_fonts(self):
        """配置Matplotlib支持中文显示"""
        # 设置字体列表，增加更多备选字体提高兼容性
        plt.rcParams["font.family"] = [
            "SimHei",         # 通用黑体
            "WenQuanYi Micro Hei",
            "Heiti TC",       # macOS黑体
            "Microsoft YaHei",# Windows雅黑
            "Arial Unicode MS"# 跨平台备选
        ]
        # 解决负号显示问题
        plt.rcParams["axes.unicode_minus"] = False

    def clear_chart(self):
        """清除当前图表"""
        self.figure.clear()
        self.canvas.draw()
        self.current_chart = None

    def plot_chart(self, data, chart_type, title, x_label=None, y_label=None):
        """
        绘制图表

        参数:
            data: 图表数据，由prepare_chart_data准备
            chart_type: 图表类型 (bar, line, pie, scatter, hist等)
            title: 图表标题
            x_label: x轴标签
            y_label: y轴标签
        """
        try:
            self.clear_chart()

            if not data or not isinstance(data, dict):
                self._show_error("无效的图表数据")
                return

            ax = self.figure.add_subplot(111)
            ax.set_title(title)

            if x_label:
                ax.set_xlabel(x_label)
            if y_label:
                ax.set_ylabel(y_label)

            # 根据图表类型绘制
            if chart_type == 'bar':
                ax.bar(data['x'], data['y'])

            elif chart_type == 'line':
                ax.plot(data['x'], data['y'], marker='o')

            elif chart_type == 'pie':
                ax.pie(data['values'], labels=data['labels'], autopct='%1.1f%%')
                ax.axis('equal')  # 保证饼图是圆的

            elif chart_type == 'scatter':
                ax.scatter(data['x'], data['y'], alpha=0.6)

            elif chart_type == 'hist':
                ax.hist(data['values'], bins=data.get('bins', 10))

            else:
                self._show_error(f"不支持的图表类型: {chart_type}")
                return

            self.figure.tight_layout()
            self.canvas.draw()
            self.current_chart = chart_type

        except Exception as e:
            self._show_error(f"绘制图表失败: {str(e)}")

    def _show_error(self, message):
        """显示错误信息在图表区域"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()