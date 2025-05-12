import pandas as pd
import numpy as np

def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str)
        print(f"Đã đọc file {csv_path} với {len(df)} hàng và {len(df.columns)} cột.")

        invalid_rows = df[df['url_length'] == 'url_length']
        if not invalid_rows.empty:
            print(f"Tìm thấy {len(invalid_rows)} hàng không hợp lệ, đang xóa...")
            df = df[df['url_length'] != 'url_length']

        for col in df.columns:
            if col == 'URL':
                df[col] = df[col].astype(str) 
            elif col == 'Label':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int if col != 'letter_to_number_ratio' else float)

        print("Kiểu dữ liệu sau khi chuyển đổi:")
        print(df.dtypes)

        return df
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        raise

def preprocess_data(df):
    if 'URL' not in df.columns or 'Label' not in df.columns:
        raise ValueError("File CSV thiếu cột 'URL' hoặc 'Label'")
    X = df.drop(columns=['URL', 'Label'])
    y = df['Label']
    print(f"Số đặc trưng: {X.shape[1]}")
    return X, y