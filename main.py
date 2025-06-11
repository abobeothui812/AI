# main.py
import osmnx as ox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib # Để tải model và preprocessor

from config import DEFAULT_TARGET_HOUR, DEFAULT_TARGET_DAY_NUMERIC
from data_loader import load_road_network, load_taxi_zones, load_taxi_trip_data
from data_processor import (
    filter_taxi_zones_by_borough,
    initial_trip_data_cleaning,
    filter_trips_by_location_ids,
    calculate_median_speed_by_time
    # create_ml_training_data # Không cần thiết ở main.py nữa trừ khi bạn muốn tạo fallback data
)
from routing_utils import (
    add_travel_times_to_graph,
    calculate_eta_for_route
)

# Hàm get_user_inputs giữ nguyên như trước
def get_user_inputs(G_graph):
    # ... (Nội dung hàm giữ nguyên) ...
    while True:
        try:
            origin_address = input("Nhập địa chỉ điểm xuất phát ở Manhattan (ví dụ: 'Times Square, New York'): ")
            origin_lat, origin_lon = ox.geocode(origin_address)
            origin_node = ox.nearest_nodes(G_graph, X=origin_lon, Y=origin_lat)
            print(f"Tìm thấy Node xuất phát gần nhất: {origin_node} cho địa chỉ '{origin_address}'")
            break
        except Exception as e:
            print(f"Lỗi geocoding địa chỉ xuất phát: {e}. Vui lòng thử lại.")
    while True:
        try:
            destination_address = input("Nhập địa chỉ điểm đến ở Manhattan (ví dụ: 'Wall Street, New York'): ")
            dest_lat, dest_lon = ox.geocode(destination_address)
            destination_node = ox.nearest_nodes(G_graph, X=dest_lon, Y=dest_lat)
            print(f"Tìm thấy Node đích gần nhất: {destination_node} cho địa chỉ '{destination_address}'")
            break
        except Exception as e:
            print(f"Lỗi geocoding địa chỉ đích: {e}. Vui lòng thử lại.")
    while True:
        try:
            hour_str = input(f"Nhập giờ khởi hành (0-23, mặc định {DEFAULT_TARGET_HOUR}): ")
            if not hour_str: 
                hour = DEFAULT_TARGET_HOUR
                break
            hour = int(hour_str)
            if 0 <= hour <= 23:
                break
            else:
                print("Giờ không hợp lệ. Vui lòng nhập số từ 0 đến 23.")
        except ValueError:
            print("Vui lòng nhập một số nguyên cho giờ.")
    while True:
        try:
            day_str = input(f"Nhập ngày khởi hành (0=Thứ Hai, ..., 6=Chủ Nhật, mặc định {DEFAULT_TARGET_DAY_NUMERIC}): ")
            if not day_str:
                day = DEFAULT_TARGET_DAY_NUMERIC
                break
            day = int(day_str)
            if 0 <= day <= 6:
                break
            else:
                print("Ngày không hợp lệ. Vui lòng nhập số từ 0 đến 6.")
        except ValueError:
            print("Vui lòng nhập một số nguyên cho ngày.")
    return origin_node, destination_node, hour, day


def main():
    print("Bắt đầu quy trình tính toán ETA...")

    # --- 1. Tải dữ liệu ---
    G_manhattan = load_road_network()
    taxi_zones_gdf = load_taxi_zones() 
    raw_taxi_trips_df = load_taxi_trip_data()

    if G_manhattan is None or taxi_zones_gdf is None or raw_taxi_trips_df is None:
        print("Lỗi tải dữ liệu đầu vào. Kết thúc chương trình.")
        return

    origin_node, destination_node, target_hour, target_day_numeric = get_user_inputs(G_manhattan)
    if origin_node is None or destination_node is None:
        print("Không thể xác định điểm đầu hoặc cuối từ địa chỉ. Kết thúc chương trình.")
        return

    # --- 2. Xử lý dữ liệu & Chuẩn bị dữ liệu cho fallback ---
    # Vẫn cần tính fallback_median_speed_by_hour cho hàm add_travel_times_to_graph
    _, manhattan_location_ids = filter_taxi_zones_by_borough(taxi_zones_gdf)
    cleaned_taxi_trips_df = initial_trip_data_cleaning(raw_taxi_trips_df)
    manhattan_trips_df = pd.DataFrame() 
    if cleaned_taxi_trips_df is not None and not cleaned_taxi_trips_df.empty and manhattan_location_ids:
        manhattan_trips_df = filter_trips_by_location_ids(cleaned_taxi_trips_df, manhattan_location_ids)
    
    fallback_median_speed_by_hour = pd.Series(dtype='float64') 
    if not manhattan_trips_df.empty:
        median_speed_by_hour_for_fallback, _ = calculate_median_speed_by_time(manhattan_trips_df)
        if not median_speed_by_hour_for_fallback.empty:
             fallback_median_speed_by_hour = median_speed_by_hour_for_fallback
        else:
            print("Cảnh báo: Không tính được fallback_median_speed_by_hour.")
    else:
        print("Cảnh báo: Không có dữ liệu manhattan_trips_df để tính fallback_median_speed_by_hour.")
    
    if fallback_median_speed_by_hour.empty: 
        print("CẢNH BÁO: fallback_median_speed_by_hour rỗng. Tốc độ fallback sẽ là giá trị mặc định.")
        fallback_median_speed_by_hour = pd.Series([10.0]*24, index=range(24))


    # --- Tải Mô hình ML và Preprocessor ĐÃ HUẤN LUYỆN ---
    print("\n--- Tải Mô hình ML và Preprocessor ---")
    ml_model = None
    ml_preprocessor = None
    try:
        ml_model = joblib.load('trained_rf_model.joblib')
        ml_preprocessor = joblib.load('data_preprocessor.joblib')
        print("Tải mô hình và preprocessor thành công.")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file model ('trained_rf_model.joblib') hoặc preprocessor ('data_preprocessor.joblib').")
        print("Vui lòng chạy script 'train_model.py' để huấn luyện và lưu mô hình trước khi chạy file này.")
        return # Kết thúc nếu không có mô hình

    if ml_model is None or ml_preprocessor is None:
        print("Không tải được mô hình ML hoặc preprocessor. Kết thúc.")
        return

    # --- 3. Tính toán ETA sử dụng Mô hình ML ---
    add_travel_times_to_graph(
        G_manhattan, 
        target_hour, 
        target_day_numeric,
        ml_model,
        ml_preprocessor,
        taxi_zones_gdf,
        fallback_median_speed_by_hour
    )
    G_with_times = G_manhattan 

    print(f"\nSẽ tính ETA cho thời điểm: {target_hour} giờ, ngày thứ {target_day_numeric} trong tuần.")
    print(f"Từ Node ID: {origin_node} đến Node ID: {destination_node}")

    route, eta_minutes = calculate_eta_for_route(G_with_times, origin_node, destination_node)

    if route and eta_minutes is not None and not (isinstance(eta_minutes, float) and (pd.isna(eta_minutes) or np.isinf(eta_minutes))):
        print(f"\n--- KẾT QUẢ ETA CUỐI CÙNG (sử dụng ML) ---")
        print(f"Lộ trình tìm được có {len(route)} nút.")
        print(f"ETA dự kiến: {eta_minutes:.2f} phút.")
        try:
            print("\nĐang vẽ lộ trình...")
            fig, ax = ox.plot_graph_route(
                G_with_times, route, route_color="b", route_linewidth=4,      
                node_size=0, bgcolor="w", show=True, close=False,            
                figsize=(10,10), dpi=100                 
            )
            ax.set_title(f"Lộ trình (ML) từ {origin_node} đến {destination_node}\nETA: {eta_minutes:.2f} phút ({target_hour}h, Ngày {target_day_numeric})", fontsize=15)
            plt.show() 
            print("Hoàn tất vẽ lộ trình.")
        except Exception as e_plot:
            print(f"Lỗi khi vẽ lộ trình: {e_plot}")
    else:
        print("Không thể tính toán ETA hợp lệ cho các điểm đã chọn với mô hình ML.")

    print("\nQuy trình tính toán ETA (sử dụng ML) kết thúc.")

if __name__ == '__main__':
    main()