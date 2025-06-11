import matplotlib.pyplot as plt
import pandas as pd # Đảm bảo pandas đã được import

# Giả sử df_manhattan_trips_cleaned_for_speed đã được tạo từ bước trước
# và chứa cột 'tpep_pickup_datetime' và 'average_speed_mph'

# Kiểm tra xem DataFrame có tồn tại và có dữ liệu không
if 'df_manhattan_trips_cleaned_for_speed' in locals() and not df_manhattan_trips_cleaned_for_speed.empty:
    df_analysis = df_manhattan_trips_cleaned_for_speed.copy()

    # 1. Trích xuất Giờ trong ngày và Ngày trong tuần từ thời gian đón khách
    # Đảm bảo cột thời gian là kiểu datetime
    df_analysis['tpep_pickup_datetime'] = pd.to_datetime(df_analysis['tpep_pickup_datetime'])
    
    df_analysis['pickup_hour'] = df_analysis['tpep_pickup_datetime'].dt.hour
    df_analysis['pickup_day_of_week'] = df_analysis['tpep_pickup_datetime'].dt.dayofweek
    # Ghi chú: dt.dayofweek trả về Thứ Hai=0, Chủ Nhật=6

    # 2. Tính tốc độ trung bình theo Giờ trong ngày
    avg_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].mean()

    print("\nTốc độ trung bình (mph) theo Giờ trong ngày:")
    print(avg_speed_by_hour)

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    avg_speed_by_hour.plot(kind='line', marker='o')
    plt.title('Tốc độ Trung bình các Chuyến đi trong Manhattan theo Giờ trong Ngày')
    plt.xlabel('Giờ trong Ngày (0-23)')
    plt.ylabel('Tốc độ Trung bình (mph)')
    plt.xticks(range(0, 24)) # Đảm bảo tất cả các giờ đều được hiển thị
    plt.grid(True)
    plt.show()

    # 3. Tính tốc độ trung bình theo Ngày trong tuần
    avg_speed_by_day_of_week = df_analysis.groupby('pickup_day_of_week')['average_speed_mph'].mean()
    # Đổi tên chỉ mục cho dễ đọc
    days = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    avg_speed_by_day_of_week.index = avg_speed_by_day_of_week.index.map(lambda x: days[x])


    print("\nTốc độ trung bình (mph) theo Ngày trong tuần:")
    print(avg_speed_by_day_of_week)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    avg_speed_by_day_of_week.plot(kind='bar', color='skyblue')
    plt.title('Tốc độ Trung bình các Chuyến đi trong Manhattan theo Ngày trong Tuần')
    plt.xlabel('Ngày trong Tuần')
    plt.ylabel('Tốc độ Trung bình (mph)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

else:
    print("DataFrame 'df_manhattan_trips_cleaned_for_speed' chưa được tạo hoặc rỗng. Hãy kiểm tra các bước trước.")