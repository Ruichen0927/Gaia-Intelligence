import pandas as pd
import os
from pathlib import Path
from typing import Dict

def run(config: dict, project_root: Path) -> dict:
    try:
        print("\n===== 启动工具: gr_mean =====")
        paths = config["paths"]
        input_file = paths["input_file"]
        output_directory = paths["output_directory"]
        output_file = os.path.join(output_directory, "gr_mean_output.csv")

        # 读取数据
        df = pd.read_csv(input_file)
        if 'GR' not in df.columns:
            return {"success": False, "message": "输入文件中未找到 'GR' 列。", "output_files": []}

        # 计算GR均值
        gr_mean = df['GR'].mean()
        # 保存结果
        result_df = pd.DataFrame({'GR_mean': [gr_mean]})
        result_df.to_csv(output_file, index=False)

        return {"success": True, "message": "GR均值计算成功。", "output_files": [output_file]}
    except Exception as e:
        return {"success": False, "message": f"发生错误: {str(e)}", "output_files": []}