# data_processor.py
import pandas as pd
from config import (
    TARGET_BOROUGH, MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES,
    MIN_TRIP_DISTANCE_MILES, MIN_AVG_SPEED_MPH, MAX_AVG_SPEED_MPH
)

def filter_taxi_zones_by_borough(taxi_zones_gdf, borough_name=TARGET_BOROUGH):
    """Lọc các khu vực taxi theo quận."""
    if taxi_zones_gdf is None:
        return None, []
    manhattan_zones = taxi_zones_gdf[taxi_zones_gdf['borough'] == borough_name]
    manhattan_location_ids = manhattan_zones['LocationID'].tolist()
    print(f"Số lượng khu vực taxi ở {borough_name}: {len(manhattan_zones)}")
    return manhattan_zones, manhattan_location_ids


def initial_trip_data_cleaning(df_taxi):
    """Làm sạch dữ liệu chuyến đi ban đầu và tính toán thời gian/tốc độ."""
    if df_taxi is None:
        return None

    df_cleaned = df_taxi.copy()
    df_cleaned['trip_duration_seconds'] = \
        (df_cleaned['tpep_dropoff_datetime'] - df_cleaned['tpep_pickup_datetime']).dt.total_seconds()
    df_cleaned['trip_duration_minutes'] = df_cleaned['trip_duration_seconds'] / 60

    # Lọc cơ bản
    df_filtered = df_cleaned[
        (df_cleaned['trip_duration_minutes'] >= MIN_TRIP_DURATION_MINUTES) &
        (df_cleaned['trip_duration_minutes'] <= MAX_TRIP_DURATION_MINUTES) &
        (df_cleaned['trip_distance'] >= MIN_TRIP_DISTANCE_MILES)
        ].copy()  # Sử dụng .copy()

    # Tính tốc độ trung bình (mph)
    # Đảm bảo trip_duration_seconds > 0 để tránh chia cho 0
    valid_duration_mask = df_filtered['trip_duration_seconds'] > 0
    df_filtered.loc[valid_duration_mask, 'average_speed_mph'] = \
        df_filtered.loc[valid_duration_mask, 'trip_distance'] / \
        (df_filtered.loc[valid_duration_mask, 'trip_duration_seconds'] / 3600)

    # Lọc tốc độ bất thường
    if 'average_speed_mph' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['average_speed_mph'] >= MIN_AVG_SPEED_MPH) &
            (df_filtered['average_speed_mph'] <= MAX_AVG_SPEED_MPH)
            ]
    else:  # Nếu không có chuyến nào thỏa mãn để tính average_speed_mph
        print("Cảnh báo: Không có chuyến đi nào đủ điều kiện để tính average_speed_mph sau lọc cơ bản.")
        # Tạo cột với giá trị NaN để tránh lỗi sau này nếu df_filtered không rỗng
        if not df_filtered.empty:
            df_filtered['average_speed_mph'] = pd.NA

    print(f"Số chuyến đi sau khi lọc cơ bản và lọc tốc độ: {len(df_filtered)}")
    return df_filtered


def filter_trips_by_location_ids(df_taxi_filtered, location_ids):
    """Lọc các chuyến đi có điểm đón VÀ trả trong danh sách LocationID đã cho."""
    if df_taxi_filtered is None or not location_ids:
        return pd.DataFrame()  # Trả về DataFrame rỗng nếu đầu vào không hợp lệ

    # Đảm bảo kiểu dữ liệu của ID là int để khớp
    df_filtered = df_taxi_filtered.copy()
    if 'PULocationID' in df_filtered.columns:
        df_filtered['PULocationID'] = df_filtered['PULocationID'].astype(int)
    if 'DOLocationID' in df_filtered.columns:
        df_filtered['DOLocationID'] = df_filtered['DOLocationID'].astype(int)

    df_borough_trips = df_filtered[
        df_filtered['PULocationID'].isin(location_ids) &
        df_filtered['DOLocationID'].isin(location_ids)
        ]
    print(f"Số chuyến đi hoàn toàn trong các khu vực đã chọn: {len(df_borough_trips)}")
    return df_borough_trips


def calculate_median_speed_by_time(df_borough_trips):
    """Tính tốc độ trung vị theo giờ và ngày trong tuần."""
    if df_borough_trips is None or df_borough_trips.empty or 'average_speed_mph' not in df_borough_trips.columns:
        print(
            "Cảnh báo: Không có dữ liệu chuyến đi của quận hoặc thiếu cột 'average_speed_mph' để tính tốc độ trung vị.")
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')

    df_analysis = df_borough_trips.copy()
    df_analysis['pickup_hour'] = df_analysis['tpep_pickup_datetime'].dt.hour
    df_analysis['pickup_day_of_week'] = df_analysis['tpep_pickup_datetime'].dt.dayofweek  # Monday=0, Sunday=6

    median_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].median()
    median_speed_by_day_of_week = df_analysis.groupby('pickup_day_of_week')['average_speed_mph'].median()

    # Đổi tên index cho ngày trong tuần
    days = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    median_speed_by_day_of_week.index = median_speed_by_day_of_week.index.map(lambda x: days[x] if x < len(days) else x)

    return median_speed_by_hour, median_speed_by_day_of_week


if __name__ == '__main__':
    # Chạy thử các hàm xử lý (cần file dữ liệu đã tải)
    from data_loader import load_taxi_zones, load_taxi_trip_data

    zones_gdf_test = load_taxi_zones()
    _, manhattan_ids_test = filter_taxi_zones_by_borough(zones_gdf_test)

    raw_taxi_data_test = load_taxi_trip_data()
    cleaned_taxi_data_test = initial_trip_data_cleaning(raw_taxi_data_test)

    if cleaned_taxi_data_test is not None and manhattan_ids_test:
        manhattan_trips_test = filter_trips_by_location_ids(cleaned_taxi_data_test, manhattan_ids_test)
        if not manhattan_trips_test.empty:
            median_hr_test, median_dow_test = calculate_median_speed_by_time(manhattan_trips_test)
            print("\nMedian speed by hour (test):\n", median_hr_test)
            print("\nMedian speed by day of week (test):\n", median_dow_test)


def create_ml_training_data(df_borough_trips):
    """
    Tạo dữ liệu huấn luyện cho mô hình ML dự đoán tốc độ trung vị theo Zone, Giờ, Ngày.
    Features: PULocationID, pickup_hour, pickup_day_of_week
    Target: median_average_speed_mph
    """
    if df_borough_trips is None or df_borough_trips.empty or 'average_speed_mph' not in df_borough_trips.columns:
        print("Cảnh báo: Không có dữ liệu chuyến đi của quận hoặc thiếu cột 'average_speed_mph' để tạo dữ liệu ML.")
        return pd.DataFrame()  # Trả về DataFrame rỗng

    df_ml = df_borough_trips.copy()

    # Đảm bảo các cột thời gian đã được trích xuất
    if 'pickup_hour' not in df_ml.columns:
        df_ml['pickup_hour'] = pd.to_datetime(df_ml['tpep_pickup_datetime']).dt.hour
    if 'pickup_day_of_week' not in df_ml.columns:  # Monday=0, Sunday=6
        df_ml['pickup_day_of_week'] = pd.to_datetime(df_ml['tpep_pickup_datetime']).dt.dayofweek

    # Tính tốc độ trung vị cho mỗi nhóm (ZoneID, Giờ, Ngày trong tuần)
    # Đây sẽ là target 'y' của chúng ta
    df_zone_hourly_daily_speed = df_ml.groupby(
        ['PULocationID', 'pickup_hour', 'pickup_day_of_week']
    )['average_speed_mph'].median().reset_index()

    # Đổi tên cột target cho rõ ràng
    df_zone_hourly_daily_speed.rename(columns={'average_speed_mph': 'target_median_speed_mph'}, inplace=True)

    # Loại bỏ các dòng có target là NaN (nếu có, ví dụ do nhóm đó không có dữ liệu average_speed_mph hợp lệ)
    df_zone_hourly_daily_speed.dropna(subset=['target_median_speed_mph'], inplace=True)

    print(f"Đã tạo được {len(df_zone_hourly_daily_speed)} mẫu dữ liệu huấn luyện ML.")

    # Các cột PULocationID, pickup_hour, pickup_day_of_week sẽ là features X
    # Cột target_median_speed_mph sẽ là target y
    return df_zone_hourly_daily_speed


if __name__ == '__main__':
    # ... (phần test cũ) ...
    # Thêm test cho hàm mới
    from data_loader import load_taxi_zones, load_taxi_trip_data  # Cần import lại nếu chạy file này độc lập

    zones_gdf_test = load_taxi_zones()
    if zones_gdf_test is not None:
        _, manhattan_ids_test = filter_taxi_zones_by_borough(zones_gdf_test)

        raw_taxi_data_test = load_taxi_trip_data()
        if raw_taxi_data_test is not None:
            cleaned_taxi_data_test = initial_trip_data_cleaning(raw_taxi_data_test)

            if cleaned_taxi_data_test is not None and not cleaned_taxi_data_test.empty and manhattan_ids_test:
                manhattan_trips_test = filter_trips_by_location_ids(cleaned_taxi_data_test, manhattan_ids_test)
                if not manhattan_trips_test.empty:
                    # median_hr_test, median_dow_test = calculate_median_speed_by_time(manhattan_trips_test)
                    # print("\nMedian speed by hour (test):\n", median_hr_test)
                    # print("\nMedian speed by day of week (test):\n", median_dow_test)

                    ml_data_test = create_ml_training_data(manhattan_trips_test)
                    if not ml_data_test.empty:
                        print("\nDữ liệu huấn luyện ML (5 dòng đầu):")
                        print(ml_data_test.head())
                        print("\nThông tin dữ liệu ML:")
                        ml_data_test.info()
                        print("\nThống kê mô tả target:")
                        print(ml_data_test['target_median_speed_mph'].describe())

