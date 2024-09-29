import pandas as pd
import os


def process_csv_file(file_path):
    # 檢查文件是否存在
    if not os.path.exists(file_path):
        print(f"錯誤：文件 '{file_path}' 不存在。")
        return None

    # 檢查文件是否為 CSV
    if not file_path.lower().endswith(".csv"):
        print(f"錯誤：文件 '{file_path}' 不是 CSV 文件。")
        return None

    try:
        # 讀取 CSV 文件
        df = pd.read_csv(file_path)
        print(f"成功讀取文件 '{file_path}'")
        print(f"文件包含 {len(df)} 行和 {len(df.columns)} 列")

        # 在這裡添加您的數據處理邏輯
        # 例如：
        # processed_data = your_processing_function(df)

        return df
    except Exception as e:
        print(f"處理文件 '{file_path}' 時發生錯誤：{str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    file_path = input("請輸入 CSV 文件的路徑：")
    result = process_csv_file(file_path)

    if result is not None:
        # 在這裡添加進一步的數據處理或分析代碼
        print(result.head())  # 顯示前幾行數據作為示例
