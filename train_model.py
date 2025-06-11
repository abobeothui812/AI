# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import các hàm cần thiết từ các module khác
from config import YELLOW_TAXI_DATA_FILE # Để có đường dẫn file dữ liệu taxi
from data_loader import load_taxi_zones, load_taxi_trip_data
from data_processor import (
    filter_taxi_zones_by_borough,
    initial_trip_data_cleaning,
    filter_trips_by_location_ids,
    create_ml_training_data # Hàm quan trọng để tạo dữ liệu ML
)

def train_and_save_model():
    """
    Hàm chính để tải dữ liệu, xử lý, huấn luyện mô hình ML,
    đánh giá và lưu mô hình cùng preprocessor.
    """
    print("Bắt đầu quy trình huấn luyện mô hình dự đoán tốc độ...")

    # --- 1. Tải dữ liệu cơ bản ---
    # (Tương tự như trong main.py, nhưng chỉ lấy những gì cần cho việc tạo ml_training_df)
    print("\n--- Bước 1: Tải dữ liệu ---")
    taxi_zones_gdf = load_taxi_zones()
    raw_taxi_trips_df = load_taxi_trip_data(YELLOW_TAXI_DATA_FILE) # Sử dụng đường dẫn từ config

    if taxi_zones_gdf is None or raw_taxi_trips_df is None:
        print("Lỗi tải dữ liệu đầu vào (taxi zones hoặc raw taxi trips). Kết thúc huấn luyện.")
        return

    # --- 2. Xử lý dữ liệu để có manhattan_trips_df ---
    print("\n--- Bước 2: Xử lý dữ liệu để lấy các chuyến đi trong Manhattan ---")
    _, manhattan_location_ids = filter_taxi_zones_by_borough(taxi_zones_gdf) # Mặc định là Manhattan từ config
    if not manhattan_location_ids:
        print("Không tìm thấy LocationID nào cho Manhattan. Kết thúc huấn luyện.")
        return
        
    cleaned_taxi_trips_df = initial_trip_data_cleaning(raw_taxi_trips_df)
    if cleaned_taxi_trips_df is None or cleaned_taxi_trips_df.empty:
        print("Không có dữ liệu taxi sau khi làm sạch ban đầu. Kết thúc huấn luyện.")
        return

    manhattan_trips_df = filter_trips_by_location_ids(cleaned_taxi_trips_df, manhattan_location_ids)
    if manhattan_trips_df.empty:
        print("Không có chuyến đi nào hoàn toàn trong Manhattan để huấn luyện. Kết thúc huấn luyện.")
        return
    print(f"Đã lọc được {len(manhattan_trips_df)} chuyến đi trong Manhattan.")

    # --- 3. Tạo Dữ liệu Huấn luyện ML ---
    print("\n--- Bước 3: Tạo Dữ liệu Huấn luyện cho Mô hình ML ---")
    ml_training_df = create_ml_training_data(manhattan_trips_df)

    if ml_training_df.empty:
        print("Không thể tạo dữ liệu huấn luyện ML. Kết thúc huấn luyện.")
        return
    
    print(f"Dữ liệu huấn luyện ML được tạo với {len(ml_training_df)} mẫu.")
    print("5 dòng đầu của dữ liệu huấn luyện ML:")
    print(ml_training_df.head())

    # --- 4. Tiền xử lý Dữ liệu ---
    print("\n--- Bước 4: Tiền xử lý Dữ liệu cho Học máy ---")
    X = ml_training_df[['PULocationID', 'pickup_hour', 'pickup_day_of_week']]
    y = ml_training_df['target_median_speed_mph']

    categorical_features = ['PULocationID']
    # numerical_features = ['pickup_hour', 'pickup_day_of_week'] # Có thể scale nếu muốn

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' 
    )

    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Kích thước tập huấn luyện X: {X_train.shape}")
    print(f"Kích thước tập kiểm thử X: {X_test.shape}")

    # Fit preprocessor trên X_train và transform cả X_train, X_test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Tiền xử lý dữ liệu hoàn tất.")

    # --- 5. Huấn luyện Mô hình ---
    print("\n--- Bước 5: Huấn luyện Mô hình (RandomForestRegressor) ---")
    rf_model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, oob_score=True,
        max_depth=20, min_samples_split=5, min_samples_leaf=2
    )
    
    print("Đang huấn luyện mô hình...")
    rf_model.fit(X_train_processed, y_train)
    print("Huấn luyện mô hình hoàn tất.")
    if hasattr(rf_model, 'oob_score_') and rf_model.oob_score_: # Kiểm tra oob_score_ có tồn tại không
        print(f"Out-of-Bag R^2 score: {rf_model.oob_score_:.4f}")

    # --- 6. Đánh giá Mô hình ---
    print("\n--- Bước 6: Đánh giá Mô hình ---")
    y_pred_train = rf_model.predict(X_train_processed)
    y_pred_test = rf_model.predict(X_test_processed)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("\nKết quả đánh giá trên tập Huấn luyện:")
    print(f"  Mean Absolute Error (MAE): {mae_train:.2f} mph")
    print(f"  R-squared (R2): {r2_train:.4f}")
    print("\nKết quả đánh giá trên tập Kiểm thử:")
    print(f"  Mean Absolute Error (MAE): {mae_test:.2f} mph")
    print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f} mph")
    print(f"  R-squared (R2): {r2_test:.4f}")

    # --- 7. Lưu Mô hình và Preprocessor ---
    print("\n--- Bước 7: Lưu Mô hình và Preprocessor ---")
    model_filename = 'trained_rf_model.joblib'
    preprocessor_filename = 'data_preprocessor.joblib'
    
    joblib.dump(rf_model, model_filename)
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Đã lưu mô hình vào file: {model_filename}")
    print(f"Đã lưu preprocessor vào file: {preprocessor_filename}")

    print("\nQuy trình huấn luyện và lưu mô hình hoàn tất.")

if __name__ == '__main__':
    train_and_save_model()