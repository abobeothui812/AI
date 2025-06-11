import geopandas as gpd
import pandas as pd # Đảm bảo pandas cũng được import
import sys
import matplotlib.pyplot as plt
# Đường dẫn đến file shapefile của Taxi Zones (thay đổi nếu cần)
# Ví dụ: nếu bạn lưu trong thư mục con 'nyc_taxi_zones' và file tên là 'taxi_zones.shp'
try:
    taxi_zones_gdf = gpd.read_file("nyc_taxi_zones/taxi_zones.shp") # Hoặc đường dẫn chính xác của bạn
    print("Đọc file Shapefile Khu vực Taxi thành công!")

    print("\n5 dòng đầu của dữ liệu Khu vực Taxi:")
    print(taxi_zones_gdf.head())

    print("\nThông tin các cột:")
    print(taxi_zones_gdf.info())
    # Các cột quan trọng thường là:
    # - LocationID: ID của khu vực (số nguyên)
    # - zone: Tên của khu vực (text)
    # - borough: Tên quận (text)
    # - geometry: Đối tượng hình học của khu vực

    # Chuyển đổi LocationID sang kiểu số nguyên nếu cần (đôi khi đọc vào có thể là text)
    if 'LocationID' in taxi_zones_gdf.columns:
        taxi_zones_gdf['LocationID'] = taxi_zones_gdf['LocationID'].astype(int)

    # Lọc ra các khu vực thuộc Manhattan
    manhattan_zones_gdf = taxi_zones_gdf[taxi_zones_gdf['borough'] == "Manhattan"]
    print(f"\nSố lượng khu vực taxi ở Manhattan: {len(manhattan_zones_gdf)}")

    # Lấy danh sách các LocationID thuộc Manhattan
    manhattan_location_ids = manhattan_zones_gdf['LocationID'].tolist()
    # print("\nDanh sách LocationID ở Manhattan (một vài ID đầu):")
    # print(manhattan_location_ids[:10])

except FileNotFoundError:
    print("LỖI: Không tìm thấy file Shapefile Khu vực Taxi. Hãy kiểm tra đường dẫn.")
except Exception as e:
    print(f"Có lỗi xảy ra khi đọc Shapefile: {e}")

# Giả sử df_taxi_filtered đã được tạo và lọc cơ bản từ bước trước
# (đã tính trip_duration_minutes, trip_distance > 0, ...)
# Và manhattan_location_ids đã được tạo ở Bước 4b

# Đường dẫn file taxi parquet (để chạy lại nếu cần)
taxi_file_path = "yellow_tripdata_2025-01.parquet" # Thay đổi nếu tên file của bạn khác

try:
    # Đọc lại dữ liệu taxi nếu chưa có trong bộ nhớ hoặc muốn đảm bảo tính nhất quán
    df_taxi_full_month = pd.read_parquet(taxi_file_path)
    df_taxi_full_month['tpep_pickup_datetime'] = pd.to_datetime(df_taxi_full_month['tpep_pickup_datetime'])
    df_taxi_full_month['tpep_dropoff_datetime'] = pd.to_datetime(df_taxi_full_month['tpep_dropoff_datetime'])
    df_taxi_full_month['trip_duration_seconds'] = (df_taxi_full_month['tpep_dropoff_datetime'] - df_taxi_full_month['tpep_pickup_datetime']).dt.total_seconds()
    df_taxi_full_month['trip_duration_minutes'] = df_taxi_full_month['trip_duration_seconds'] / 60

    # Lọc cơ bản ban đầu (như đã làm)
    df_taxi_filtered_initial = df_taxi_full_month[
        (df_taxi_full_month['trip_duration_minutes'] > 1) &
        (df_taxi_full_month['trip_duration_minutes'] < 120) &
        (df_taxi_full_month['trip_distance'] > 0)
    ].copy() # Thêm .copy() để tránh SettingWithCopyWarning

    print(f"Tổng số chuyến đi trong file sau lọc cơ bản: {len(df_taxi_filtered_initial)}")

    # Đảm bảo manhattan_location_ids đã tồn tại
    if 'manhattan_location_ids' in locals() or 'manhattan_location_ids' in globals():
        # Lọc các chuyến đi có cả điểm đón (PULocationID) VÀ điểm trả (DOLocationID) đều nằm trong Manhattan
        # Chuyển đổi kiểu dữ liệu của PULocationID và DOLocationID nếu cần để khớp với manhattan_location_ids
        if 'PULocationID' in df_taxi_filtered_initial.columns:
             df_taxi_filtered_initial['PULocationID'] = df_taxi_filtered_initial['PULocationID'].astype(int)
        if 'DOLocationID' in df_taxi_filtered_initial.columns:
            df_taxi_filtered_initial['DOLocationID'] = df_taxi_filtered_initial['DOLocationID'].astype(int)

        df_manhattan_trips = df_taxi_filtered_initial[
            df_taxi_filtered_initial['PULocationID'].isin(manhattan_location_ids) &
            df_taxi_filtered_initial['DOLocationID'].isin(manhattan_location_ids)
        ]
        print(f"Số chuyến đi hoàn toàn trong Manhattan: {len(df_manhattan_trips)}")

        # Nếu muốn các chuyến đi có ÍT NHẤT MỘT ĐẦU là Manhattan:
        # df_manhattan_trips_or = df_taxi_filtered_initial[
        #     df_taxi_filtered_initial['PULocationID'].isin(manhattan_location_ids) |
        #     df_taxi_filtered_initial['DOLocationID'].isin(manhattan_location_ids)
        # ]
        # print(f"Số chuyến đi có ít nhất một đầu ở Manhattan: {len(df_manhattan_trips_or)}")
    else:
        print("LỖI: Biến manhattan_location_ids chưa được tạo. Hãy chạy Bước 4b trước.")

except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file taxi {taxi_file_path}.")
except NameError as e:
    print(f"LỖI Biến: {e}. Đảm bảo bạn đã chạy các bước trước để tạo các DataFrame cần thiết.")
except Exception as e:
    print(f"Có lỗi xảy ra: {e}")

# Đảm bảo df_manhattan_trips đã được tạo và không rỗng
if 'df_manhattan_trips' in locals() and not df_manhattan_trips.empty:
    print("\nThống kê về các chuyến đi trong Manhattan:")
    print("\nThời gian chuyến đi (phút):")
    print(df_manhattan_trips['trip_duration_minutes'].describe())

    print("\nQuãng đường chuyến đi (dặm):")
    print(df_manhattan_trips['trip_distance'].describe())

    # Tính tốc độ trung bình (dặm/giờ)
    # trip_duration_seconds đã được tính ở bước trước
    # Cần đảm bảo trip_duration_seconds không bằng 0 để tránh lỗi chia cho 0
    df_manhattan_trips_cleaned_for_speed = df_manhattan_trips[df_manhattan_trips['trip_duration_seconds'] > 0].copy()
    
    # Thêm .copy() ở trên để tránh SettingWithCopyWarning khi tạo cột mới
    df_manhattan_trips_cleaned_for_speed['average_speed_mph'] = \
        df_manhattan_trips_cleaned_for_speed['trip_distance'] / (df_manhattan_trips_cleaned_for_speed['trip_duration_seconds'] / 3600)

    print("\nTốc độ trung bình (mph):")
    print(df_manhattan_trips_cleaned_for_speed['average_speed_mph'].describe())

    # Có thể xem xét loại bỏ các giá trị tốc độ quá cao hoặc quá thấp nếu cần
    # Ví dụ: tốc độ > 0 và < 100 mph
    sensible_speeds = df_manhattan_trips_cleaned_for_speed[
        (df_manhattan_trips_cleaned_for_speed['average_speed_mph'] > 0) &
        (df_manhattan_trips_cleaned_for_speed['average_speed_mph'] < 80) # Giới hạn tốc độ hợp lý
    ]['average_speed_mph']
    print("\nTốc độ trung bình (mph) sau khi lọc các giá trị bất thường:")
    print(sensible_speeds.describe())

    plt.figure(figsize=(10, 6))
    sensible_speeds.hist(bins=50)
    plt.title("Phân phối Tốc độ Trung bình các Chuyến đi trong Manhattan (mph)")
    plt.xlabel("Tốc độ Trung bình (mph)")
    plt.ylabel("Số lượng chuyến đi")
    plt.grid(axis='y', alpha=0.75)
    plt.show()


else:
    print("DataFrame df_manhattan_trips chưa được tạo hoặc rỗng. Hãy kiểm tra các bước trước.")

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

import matplotlib.pyplot as plt
import pandas as pd
import sys # Cần cho việc kiểm tra matplotlib

# Giả sử df_analysis đã được tạo từ bước trước và chứa các cột
# 'pickup_hour' và 'average_speed_mph'

if 'df_analysis' in locals() and not df_analysis.empty:
    # Đếm số lượng chuyến đi cho mỗi giờ
    trip_counts_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].count()
    # Tính lại tốc độ trung bình (để đảm bảo chúng ta đang làm việc với cùng một cơ sở)
    avg_speed_by_hour_recalc = df_analysis.groupby('pickup_hour')['average_speed_mph'].mean()
    # Tính tốc độ trung vị (median) - ít bị ảnh hưởng bởi outlier
    median_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].median()


    print("\nSố lượng chuyến đi theo Giờ trong ngày:")
    print(trip_counts_by_hour)

    print("\nTốc độ TRUNG BÌNH (mph) theo Giờ trong ngày (tính lại):")
    print(avg_speed_by_hour_recalc)

    print("\nTốc độ TRUNG VỊ (mph) theo Giờ trong ngày:")
    print(median_speed_by_hour)

    # Vẽ lại biểu đồ tốc độ trung bình và thêm biểu đồ số lượng chuyến đi
    fig, ax1 = plt.subplots(figsize=(14, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Giờ trong Ngày (0-23)')
    ax1.set_ylabel('Tốc độ Trung bình/Trung vị (mph)', color=color)
    avg_speed_by_hour_recalc.plot(kind='line', marker='o', ax=ax1, color=color, label='Tốc độ Trung bình (Mean)')
    median_speed_by_hour.plot(kind='line', marker='s', ax=ax1, color='tab:green', linestyle='--', label='Tốc độ Trung vị (Median)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(0, 24))
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()  # Tạo trục y thứ hai chia sẻ cùng trục x
    color = 'tab:red'
    ax2.set_ylabel('Số lượng chuyến đi', color=color)
    trip_counts_by_hour.plot(kind='bar', ax=ax2, color=color, alpha=0.6, label='Số lượng chuyến đi')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  # Đảm bảo mọi thứ vừa vặn
    plt.title('Phân tích Tốc độ và Số lượng chuyến đi theo Giờ trong Ngày ở Manhattan')
    plt.show()

else:
    print("DataFrame 'df_analysis' chưa được tạo hoặc rỗng. Hãy kiểm tra các bước trước.")


import pandas as pd
import osmnx as ox
import numpy as np # Sẽ dùng cho việc tránh chia cho 0

# ----- ĐẢM BẢO CÁC BIẾN NÀY ĐÃ CÓ TỪ CÁC BƯỚC TRƯỚC -----
# Ví dụ, nếu bạn cần chạy lại phần tính median_speed_by_hour:
# Giả sử df_analysis đã được tạo từ file taxi và xử lý
if 'df_analysis' not in locals() or df_analysis.empty:
    print("Biến df_analysis không tồn tại hoặc rỗng. Hãy chạy lại các bước xử lý dữ liệu taxi.")
    # Cần có code để tải lại và xử lý df_analysis ở đây nếu cần
    # Ví dụ:
    # df_analysis = df_manhattan_trips_cleaned_for_speed.copy()
    # df_analysis['tpep_pickup_datetime'] = pd.to_datetime(df_analysis['tpep_pickup_datetime'])
    # df_analysis['pickup_hour'] = df_analysis['tpep_pickup_datetime'].dt.hour
    # df_analysis['pickup_day_of_week'] = df_analysis['tpep_pickup_datetime'].dt.dayofweek
    # median_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].median()
    # median_speed_by_day_of_week = df_analysis.groupby('pickup_day_of_week')['average_speed_mph'].median()
    # days = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    # median_speed_by_day_of_week.index = median_speed_by_day_of_week.index.map(lambda x: days[x])
else: # Nếu df_analysis đã có
    median_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].median()
    # median_speed_by_day_of_week = df_analysis.groupby('pickup_day_of_week')['average_speed_mph'].median() # Có thể dùng sau
place_name = "Manhattan, New York City, New York, USA"

print(f"Đang tải dữ liệu mạng lưới đường cho {place_name}...")
# Tải dữ liệu mạng lưới đường cho loại 'drive' (đường cho xe ô tô)
# Gdf là viết tắt của Graph from place
G = ox.graph_from_place(place_name, network_type="drive")
print("Tải dữ liệu mạng lưới đường hoàn tất!")



# Giả sử G (đồ thị Manhattan) cũng đã được tải
if 'G' not in locals():
    print("Đồ thị G (Manhattan) chưa được tải. Vui lòng chạy lại bước tải bản đồ.")
    # place_name = "Manhattan, New York City, New York, USA"
    # G = ox.graph_from_place(place_name, network_type="drive")
# -------------------------------------------------------------

# 1. Chọn thời gian khởi hành mong muốn
target_hour = 10  # Ví dụ: 10 giờ sáng
target_day_numeric = 1 # Ví dụ: Thứ Ba (0=Thứ Hai, 1=Thứ Ba, ..., 6=Chủ Nhật)

# 2. Lấy tốc độ trung vị đại diện (mph) cho giờ đó
# Hiện tại, chúng ta sẽ dùng median_speed_by_hour.
# Bạn có thể mở rộng để kết hợp cả ngày trong tuần nếu muốn.
try:
    if target_hour in median_speed_by_hour.index:
        representative_speed_mph = median_speed_by_hour.loc[target_hour]
        print(f"Tốc độ trung vị đại diện cho {target_hour} giờ sáng là: {representative_speed_mph:.2f} mph")
    else:
        # Nếu không có dữ liệu cho giờ đó, lấy giá trị trung vị chung của cả ngày
        representative_speed_mph = median_speed_by_hour.median()
        print(f"Không có dữ liệu cho {target_hour} giờ, sử dụng tốc độ trung vị chung: {representative_speed_mph:.2f} mph")

    # 3. Chuyển đổi tốc độ sang m/s (vì chiều dài đoạn đường trong OSMnx là mét)
    # 1 dặm = 1609.34 mét
    # 1 giờ = 3600 giây
    # mph_to_mps = 1609.34 / 3600
    mph_to_mps = 0.44704
    representative_speed_mps = representative_speed_mph * mph_to_mps
    print(f"Tốc độ đại diện (m/s): {representative_speed_mps:.2f} m/s")

    if representative_speed_mps <= 0:
        print("CẢNH BÁO: Tốc độ đại diện <= 0. Điều này sẽ gây lỗi chia cho 0. Kiểm tra lại dữ liệu median_speed_by_hour.")
        # Có thể gán một giá trị tốc độ tối thiểu rất nhỏ để tránh lỗi, ví dụ: 1 mph
        if representative_speed_mph <=0:
            print("Gán tốc độ tối thiểu là 1mph để tiếp tục.")
            representative_speed_mps = 1 * mph_to_mps


except KeyError:
    print(f"LỖI: Không tìm thấy dữ liệu cho giờ {target_hour} trong median_speed_by_hour.")
    representative_speed_mps = None # Hoặc một giá trị mặc định an toàn
except Exception as e:
    print(f"Có lỗi xảy ra khi lấy tốc độ: {e}")
    representative_speed_mps = None

if 'G' in locals() and 'representative_speed_mps' in locals() and representative_speed_mps is not None and representative_speed_mps > 0:
    print(f"\nĐang cập nhật 'travel_time' cho các cạnh trong G với tốc độ {representative_speed_mps:.2f} m/s...")
    for u, v, data in G.edges(data=True): # Lặp trực tiếp qua các cạnh của đồ thị G
        length_m = data.get('length') # Lấy chiều dài của cạnh
        if length_m is not None:
            try:
                length_m = float(length_m) # Đảm bảo chiều dài là số thực
                data['travel_time'] = length_m / representative_speed_mps
            except ValueError:
                # print(f"Cảnh báo: Không thể chuyển đổi chiều dài '{length_m}' thành số cho cạnh ({u},{v}). Gán travel_time là vô cực.")
                data['travel_time'] = float('inf') # Hoặc xử lý lỗi theo cách khác
            except ZeroDivisionError:
                # print(f"Cảnh báo: Tốc độ là 0 cho cạnh ({u},{v}). Gán travel_time là vô cực.")
                data['travel_time'] = float('inf')
        else:
            # print(f"Cảnh báo: Cạnh ({u},{v}) thiếu thuộc tính 'length'. Gán travel_time là vô cực.")
            data['travel_time'] = float('inf') # Hoặc xử lý lỗi theo cách khác
    print("Hoàn tất cập nhật 'travel_time' cho tất cả các cạnh trong G trực tiếp.")
else:
    if 'G' not in locals():
        print("Đồ thị G chưa được tải.")
    if 'representative_speed_mps' not in locals() or representative_speed_mps is None or representative_speed_mps <= 0:
        print("Tốc độ đại diện không hợp lệ, không thể tính thời gian di chuyển.")

# ----- Bước 4 (Giữ nguyên hoặc đảm bảo bạn đã chọn node): Chọn Điểm Xuất phát và Điểm Đến -----
# Giả sử origin_node = 42423020 và destination_node = 60921167 như kết quả của bạn
if 'G' in locals():
    if 'origin_node' not in locals() or origin_node is None: # Nếu chưa có thì chọn ngẫu nhiên
        print("Chọn lại điểm ngẫu nhiên vì origin_node chưa có.")
        nodes_array = list(G.nodes())
        if len(nodes_array) >=2:
            origin_node = 42423020 # Giữ nguyên giá trị của bạn
            destination_node = 60921167 # Giữ nguyên giá trị của bạn
            # origin_node = random.choice(nodes_array)
            # destination_node = random.choice(nodes_array)
            # while destination_node == origin_node:
            #     destination_node = random.choice(nodes_array)
            print(f"Điểm xuất phát (Node ID): {origin_node}")
            print(f"Điểm đến (Node ID): {destination_node}")
        else:
            print("Không đủ nút trong đồ thị.")
            origin_node, destination_node = None, None
else:
    print("Đồ thị G chưa được tải.")
    origin_node, destination_node = None, None


# ----- Bước 5 (ĐIỀU CHỈNH): Tìm Lộ trình Ngắn nhất và Tính ETA -----
if 'G' in locals() and origin_node is not None and destination_node is not None:
    # Đảm bảo các nút tồn tại trong G
    if not G.has_node(origin_node) or not G.has_node(destination_node):
        print(f"LỖI: Nút xuất phát {origin_node} hoặc nút đích {destination_node} không tồn tại trong đồ thị G.")
    else:
        try:
            print(f"\nĐang tìm lộ trình từ {origin_node} đến {destination_node}...")
            route = ox.shortest_path(G, origin_node, destination_node, weight="travel_time")

            if route:
                print("Tìm thấy lộ trình!")
                # Sử dụng hàm tiện ích của OSMnx để lấy danh sách các thuộc tính travel_time của các cạnh trên lộ trình
                # Điều này an toàn hơn là tự lặp và truy cập G.get_edge_data
                try:
                    route_travel_times = ox.utils_graph.get_route_edge_attributes(G, route, 'travel_time')
                    total_travel_time_seconds = sum(route_travel_times)
                    
                    eta_minutes = total_travel_time_seconds / 60
                    print(f"Thời gian di chuyển dự kiến (ETA): {total_travel_time_seconds:.2f} giây ({eta_minutes:.2f} phút)")

                    # (Tùy chọn) Vẽ lộ trình lên bản đồ
                    # fig, ax = ox.plot_graph_route(G, route, route_color="r", route_linewidth=6, node_size=0, bgcolor="w")
                    # plt.show()

                except KeyError as e_route_attr:
                    print(f"Lỗi khi lấy thuộc tính 'travel_time' từ lộ trình: {e_route_attr}")
                    print("Điều này có nghĩa là một số cạnh trong lộ trình được tìm thấy vẫn thiếu thuộc tính 'travel_time' hợp lệ.")
                    print("Hãy kiểm tra kỹ lại Bước 3 (cập nhật travel_time).")
                    # In ra một vài cạnh trong lộ trình để kiểm tra
                    # for i in range(min(5, len(route) - 1)):
                    #     u, v = route[i], route[i+1]
                    #     print(f"Cạnh ({u},{v}) trong lộ trình có dữ liệu: {G.edges[u,v,0]}")


            else:
                print("Không tìm thấy lộ trình giữa hai điểm đã chọn.")

        except Exception as e:
            print(f"Lỗi khi tìm lộ trình hoặc tính ETA: {e}")
            # Cung cấp thêm thông tin nếu có thể
            if "Nodes not found in graph" in str(e):
                 print(f"LỖI: Nút xuất phát {origin_node} hoặc nút đích {destination_node} không tồn tại trong đồ thị G.")
            elif "No path found between" in str(e):
                 print(f"LỖI: Không có đường đi giữa {origin_node} và {destination_node}.")


else:
    if 'G' not in locals():
        print("Đồ thị G chưa được tải.")
    if origin_node is None or destination_node is None:
        print("Điểm xuất phát hoặc điểm đến chưa được xác định.")