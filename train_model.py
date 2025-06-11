# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import YELLOW_TAXI_DATA_FILE
from data_loader import load_road_network, load_taxi_zones, load_taxi_trip_data
from data_processor import (
    filter_taxi_zones_by_borough,
    initial_trip_data_cleaning,
    filter_trips_by_location_ids,
    create_ml_training_data,
    calculate_spatial_features # Import hàm mới
)

def train_and_save_model():
    """
    Hàm chính để tải dữ liệu, xử lý, huấn luyện mô hình ML với các đặc trưng nâng cao,
    đánh giá và lưu mô hình cùng preprocessor.
    """
    print("Bắt đầu quy trình huấn luyện mô hình dự đoán tốc độ...")

    # --- 1. Tải dữ liệu cơ bản ---
    print("\n--- Bước 1: Tải dữ liệu ---")
    G_manhattan = load_road_network()
    taxi_zones_gdf = load_taxi_zones()
    raw_taxi_trips_df = load_taxi_trip_data(YELLOW_TAXI_DATA_FILE)

    if G_manhattan is None or taxi_zones_gdf is None or raw_taxi_trips_df is None:
        print("Lỗi tải dữ liệu đầu vào. Kết thúc huấn luyện.")
        return

    # --- 2. Xử lý dữ liệu để có manhattan_trips_df ---
    print("\n--- Bước 2: Xử lý dữ liệu để lấy các chuyến đi trong Manhattan ---")
    manhattan_zones_gdf, manhattan_location_ids = filter_taxi_zones_by_borough(taxi_zones_gdf)
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

    # --- 3. Tạo Dữ liệu Huấn luyện ML với các Đặc trưng Mới ---
    print("\n--- Bước 3: Tạo Dữ liệu Huấn luyện và Đặc trưng Nâng cao ---")
    
    # Tạo dữ liệu ML với các đặc trưng thời gian tuần hoàn
    ml_training_df = create_ml_training_data(manhattan_trips_df)

    # Tính toán đặc trưng không gian cho các zone
    spatial_features_df = calculate_spatial_features(manhattan_zones_gdf, G_manhattan)

    # Hợp nhất (merge) đặc trưng không gian vào dữ liệu huấn luyện
    if not spatial_features_df.empty:
        ml_training_df = ml_training_df.merge(spatial_features_df, left_on='PULocationID', right_on='LocationID', how='left')
        ml_training_df.drop(columns=['LocationID'], inplace=True) # Bỏ cột LocationID thừa
        # Điền giá trị 0 cho các zone không có đặc trưng không gian (nếu có)
        ml_training_df['road_density'].fillna(0, inplace=True) 
    
    if ml_training_df.empty:
        print("Không thể tạo dữ liệu huấn luyện ML. Kết thúc huấn luyện.")
        return

    print("5 dòng đầu của dữ liệu huấn luyện ML sau khi hợp nhất:")
    print(ml_training_df.head())
    print("Các cột trong dữ liệu huấn luyện ML:", ml_training_df.columns.tolist())

    # --- 4. Tiền xử lý Dữ liệu ---
    print("\n--- Bước 4: Tiền xử lý Dữ liệu cho Học máy ---")
    # Các đặc trưng gốc không còn cần thiết
    features_to_drop = ['pickup_hour', 'pickup_day_of_week']
    X = ml_training_df.drop(columns=['target_median_speed_mph'] + features_to_drop)
    y = ml_training_df['target_median_speed_mph']

    # Định nghĩa các cột categorical và numerical MỚI
    categorical_features = ['PULocationID']
    # Các đặc trưng mới đều là numerical
    numerical_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'road_density']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features) # Scale các đặc trưng số
        ],
        remainder='passthrough' 
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Tiền xử lý dữ liệu hoàn tất.")

    # --- 5. Tinh chỉnh Siêu tham số & Huấn luyện Mô hình (A.2) ---
    print("\n--- Bước 5: Tinh chỉnh Siêu tham số & Huấn luyện Mô hình ---")
    
    # Định nghĩa lưới tham số cho GridSearchCV (một ví dụ nhỏ)
    # BẠN CÓ THỂ MỞ RỘNG LƯỚI NÀY, nhưng sẽ tốn nhiều thời gian huấn luyện hơn
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [20, 30],
        'min_samples_leaf': [2, 4]
    }

    # Khởi tạo GridSearchCV
    # cv=3 nghĩa là sử dụng 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1, oob_score=False), # oob_score không dùng với GridSearchCV
        param_grid=param_grid,
        cv=3,
        n_jobs=-1, # Dùng tất cả core cho GridSearchCV
        verbose=2, # In ra thông tin quá trình
        scoring='neg_mean_absolute_error' # Dùng MAE để đánh giá, nhưng GridSearchCV tối đa hóa, nên dùng neg_mae
    )

    print("Đang chạy GridSearchCV để tìm tham số tốt nhất...")
    grid_search.fit(X_train_processed, y_train)
    
    print("\nTham số tốt nhất tìm được:")
    print(grid_search.best_params_)
    
    # Lấy mô hình tốt nhất từ GridSearchCV
    best_rf_model = grid_search.best_estimator_
    print("Huấn luyện mô hình tốt nhất hoàn tất.")

    # --- 6. Đánh giá Mô hình Tốt nhất ---
    print("\n--- Bước 6: Đánh giá Mô hình Tốt nhất ---")
    y_pred_test = best_rf_model.predict(X_test_processed)
    
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print("\nKết quả đánh giá cuối cùng trên tập Kiểm thử:")
    print(f"  Mean Absolute Error (MAE): {mae_test:.2f} mph")
    print(f"  Root Mean Squared Error (RMSE): {rmse_test:.2f} mph")
    print(f"  R-squared (R2): {r2_test:.4f}")

    # --- 7. Lưu Mô hình và Preprocessor ---
    print("\n--- Bước 7: Lưu Mô hình Tốt nhất và Preprocessor ---")
    joblib.dump(best_rf_model, 'trained_rf_model.joblib')
    joblib.dump(preprocessor, 'data_preprocessor.joblib') # Lưu preprocessor đã được fit
    print("Đã lưu mô hình và preprocessor mới.")

    print("\nQuy trình huấn luyện và lưu mô hình hoàn tất.")

if __name__ == '__main__':
    train_and_save_model()
