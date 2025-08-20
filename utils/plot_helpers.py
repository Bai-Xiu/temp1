import pandas as pd
import numpy as np


def prepare_chart_data(df, chart_info):
    """转换DataFrame数据为图表所需格式（修正字段访问路径）"""
    try:
        # 基础数据校验
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("无效的DataFrame数据")
        if not chart_info or not isinstance(chart_info, dict):
            raise ValueError("无效的图表配置信息")

        # 从顶层获取基础配置
        required_top = ["chart_type", "data_prep"]
        for field in required_top:
            if field not in chart_info:
                raise ValueError(f"图表配置缺少必要顶级字段: {field}")

        chart_type = chart_info["chart_type"]
        data_prep = chart_info["data_prep"]  # 从data_prep子字典获取列配置
        if not isinstance(data_prep, dict):
            raise ValueError("chart_info.data_prep必须是字典类型")

        # 根据图表类型检查必要的列配置（从data_prep中获取）
        type_required = {
            'bar': ['x_col', 'y_col'],
            'line': ['x_col', 'y_col'],
            'scatter': ['x_col', 'y_col'],
            'pie': ['x_col', 'values'],
            'hist': ['x_col']
        }
        if chart_type not in type_required:
            raise ValueError(f"不支持的图表类型: {chart_type}（支持类型：{list(type_required.keys())}）")

        # 检查data_prep中是否包含必要字段
        for field in type_required[chart_type]:
            if field not in data_prep:
                raise ValueError(f"图表类型{chart_type}缺少必要字段: {field}（在data_prep中）")

        # 提取列名（从data_prep中）
        x_col = data_prep["x_col"]
        y_col = data_prep.get("y_col")
        values_col = data_prep.get("values")

        # 检查列是否存在于DataFrame中
        if x_col not in df.columns:
            raise ValueError(f"数据中不存在列: {x_col}")
        if y_col and y_col not in df.columns:
            raise ValueError(f"数据中不存在列: {y_col}")
        if values_col and values_col not in df.columns:
            raise ValueError(f"数据中不存在列: {values_col}")

        # 处理数据（移除空值）
        data = df.copy()
        data = data.dropna(subset=[x_col])
        if y_col:
            data = data.dropna(subset=[y_col])
        if values_col:
            data = data.dropna(subset=[values_col])

        # 按图表类型准备数据
        result = {"chart_type": chart_type}
        if chart_type in ['bar', 'line', 'scatter']:
            result['x'] = data[x_col].tolist()
            result['y'] = data[y_col].tolist()
        elif chart_type == 'pie':
            result['labels'] = data[x_col].tolist()
            result['values'] = data[values_col].tolist()
        elif chart_type == 'hist':
            result['values'] = data[x_col].tolist()
            result['bins'] = data_prep.get('bins', 10)  # 可选参数

        return result

    except Exception as e:
        print(f"准备图表数据时出错: {str(e)}")
        return None
