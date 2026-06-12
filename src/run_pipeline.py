import pandas as pd
from src.data_loader import load_config, load_raw_data
from src.preprocessing import run_baseline_pipeline

# run_pipeline script

def get_processed_dataframe() -> pd.DataFrame:
    # 1. Tải cấu hình từ config.yaml để lấy đường dẫn file dữ liệu thật
    config = load_config("configs/config.yaml")
    
    # Lấy đường dẫn file CSV từ file cấu hình (nếu không có sẽ dùng đường dẫn mặc định dưới đây)
    data_path = config.get("data", {}).get("raw_data_path", "data/raw_data/financial_fraud_detection_dataset.csv")    
    # 2. Đọc dữ liệu thật lên thành DataFrame thô
    df_raw = load_raw_data(data_path)
    
    # 3. Chạy pipeline tiền xử lý và trả về DataFrame đã sạch hoàn toàn
    df_clean = run_baseline_pipeline(df_raw)
    
    return df_clean

if __name__ == "__main__":
    # Chạy thử và hứng lấy DataFrame kết quả
    df = get_processed_dataframe()
    
    # Lúc này dữ liệu đã nằm trong biến `df`, bạn có thể dùng để train mô hình hoặc kiểm tra
    print("\n[THÀNH CÔNG] Dữ liệu đã được xử lý xong và lưu vào biến DataFrame!")
    print(f"Kích thước DataFrame hiện tại: {df.shape}")
    print(df.head())
