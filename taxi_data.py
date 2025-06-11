import pandas as pd

# Thay thế bằng tên file bạn đã tải về
# Ví dụ: nếu bạn tải file tháng 1 năm 2023 và lưu vào thư mục con 'data'
# file_path = "data/yellow_tripdata_2023-01.parquet"
# Hoặc nếu file nằm cùng thư mục với notebook của bạn:
file_path = "yellow_tripdata_2025-01.parquet" # Đảm bảo file này tồn tại

print(f"Đang đọc file: {file_path}...")
try:
    df_taxi = pd.read_parquet(file_path)
    # Nếu bạn tải file CSV, dùng: df_taxi = pd.read_csv(file_path)
    print("Đọc file hoàn tất!")

    print("\nThông tin cơ bản về DataFrame:")
    df_taxi.info()

    print("\n5 dòng dữ liệu đầu tiên:")
    print(df_taxi.head())

    print("\nCác cột có trong dữ liệu:")
    print(df_taxi.columns)

    # Một số cột quan trọng thường có (tên có thể thay đổi chút ít tùy theo tháng/năm):
    # - tpep_pickup_datetime: Thời gian đón khách
    # - tpep_dropoff_datetime: Thời gian trả khách
    # - passenger_count: Số lượng hành khách
    # - trip_distance: Quãng đường (dặm)
    # - PULocationID: ID Khu vực đón khách (Taxi Zone ID)
    # - DOLocationID: ID Khu vực trả khách (Taxi Zone ID)
    # - fare_amount: Tiền cước
    # - total_amount: Tổng tiền
    # - tip_amount: Tiền boa
    # - tolls_amount: Phí cầu đường

    # Tính toán thời gian chuyến đi (trip duration)
    # Chuyển đổi cột thời gian sang định dạng datetime nếu chưa phải
    df_taxi['tpep_pickup_datetime'] = pd.to_datetime(df_taxi['tpep_pickup_datetime'])
    df_taxi['tpep_dropoff_datetime'] = pd.to_datetime(df_taxi['tpep_dropoff_datetime'])

    df_taxi['trip_duration_seconds'] = (df_taxi['tpep_dropoff_datetime'] - df_taxi['tpep_pickup_datetime']).dt.total_seconds()
    df_taxi['trip_duration_minutes'] = df_taxi['trip_duration_seconds'] / 60

    print("\nThống kê cơ bản về thời gian chuyến đi (phút):")
    print(df_taxi['trip_duration_minutes'].describe())

    # Lọc ra các chuyến đi có thời gian hợp lý (ví dụ > 1 phút và < 2 giờ)
    # và quãng đường > 0 để loại bỏ dữ liệu nhiễu
    df_taxi_filtered = df_taxi[
        (df_taxi['trip_duration_minutes'] > 1) &
        (df_taxi['trip_duration_minutes'] < 120) &
        (df_taxi['trip_distance'] > 0)
    ]
    print(f"\nSố chuyến đi sau khi lọc cơ bản: {len(df_taxi_filtered)}")
    print("Thống kê thời gian chuyến đi sau khi lọc:")
    print(df_taxi_filtered['trip_duration_minutes'].describe())

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {file_path}. Hãy đảm bảo bạn đã tải file và đặt đúng đường dẫn.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")