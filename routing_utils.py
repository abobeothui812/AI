# routing_utils.py
import osmnx as ox
import numpy as np
import pandas as pd
import re
import geopandas as gpd # Đã import từ trước
from config import MPH_TO_MPS, ROAD_TYPE_SPEED_MODIFIERS

# --- Hàm parse_maxspeed giữ nguyên ---
def parse_maxspeed(maxspeed_str, default_speed_mph=None):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    if maxspeed_str is None:
        return default_speed_mph
    if isinstance(maxspeed_str, (int, float)): 
        return float(maxspeed_str) 
    if isinstance(maxspeed_str, list): 
        if not maxspeed_str:
            return default_speed_mph
        maxspeed_str = maxspeed_str[0]
        if isinstance(maxspeed_str, (int, float)):
             return float(maxspeed_str)
    try:
        numeric_part = re.findall(r'\d+\.?\d*', str(maxspeed_str))
        if not numeric_part:
            return default_speed_mph
        speed = float(numeric_part[0])
        if "kph" in maxspeed_str.lower() or "km/h" in maxspeed_str.lower():
            return speed / 1.60934 
        elif "mph" in maxspeed_str.lower() or not any(char.isalpha() for char in maxspeed_str.replace(numeric_part[0], "")):
            return speed
        else: 
            return default_speed_mph
    except Exception:
        return default_speed_mph

# --- Hàm get_fallback_speed_mps giữ nguyên ---
def get_fallback_speed_mps(target_hour, median_speed_by_hour_series):
    speed_mph = 10.0 
    if median_speed_by_hour_series is not None and not median_speed_by_hour_series.empty:
        if target_hour in median_speed_by_hour_series.index:
            speed_mph = median_speed_by_hour_series.loc[target_hour]
        else:
            speed_mph = median_speed_by_hour_series.median()
    if pd.isna(speed_mph): speed_mph = 10.0
    speed_mps = speed_mph * MPH_TO_MPS
    return max(speed_mps, 0.1 * MPH_TO_MPS)


def add_travel_times_to_graph(
    G, 
    target_hour, 
    target_day_numeric, 
    ml_model,                 
    ml_preprocessor,          
    taxi_zones_gdf_input, # Đổi tên để phân biệt với bản đã reproject     
    fallback_median_speed_by_hour 
):
    if G is None or ml_model is None or ml_preprocessor is None or taxi_zones_gdf_input is None:
        print("Lỗi: Thiếu đồ thị G, mô hình ML, preprocessor hoặc taxi_zones_gdf_input.")
        return G 

    print(f"\nĐang cập nhật 'travel_time' sử dụng mô hình ML cho giờ {target_hour}, ngày {target_day_numeric}...")

    # 1. Chuẩn bị gdf_edges và taxi_zones_gdf cho spatial join
    try:
        # Lấy GeoDataFrame của các cạnh từ đồ thị G
        # G.graph['crs'] thường là EPSG:4326 (WGS84 Geographic)
        current_graph_crs = G.graph.get('crs')
        if current_graph_crs is None:
            print("Cảnh báo: Đồ thị G không có thông tin CRS. Giả định là EPSG:4326.")
            current_graph_crs = "EPSG:4326"

        # Chuyển đổi G sang GeoDataFrames cho các cạnh
        # Quan trọng: không lấy nodes ở đây để tránh thay đổi G.graph['crs'] nếu G được chiếu
        _ , gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        gdf_edges = gdf_edges.set_crs(current_graph_crs, allow_override=True) # Đảm bảo gdf_edges có CRS

        # Chọn một Projected CRS phù hợp cho NYC (ví dụ: UTM Zone 18N, đơn vị mét)
        PROJECTED_CRS = "EPSG:32618"

        # Reproject gdf_edges và taxi_zones_gdf_input sang PROJECTED_CRS
        print(f"Đang reproject dữ liệu sang CRS: {PROJECTED_CRS}...")
        gdf_edges_proj = gdf_edges.to_crs(PROJECTED_CRS)
        taxi_zones_gdf_proj = taxi_zones_gdf_input.to_crs(PROJECTED_CRS)
        print("Reproject hoàn tất.")

        # Tính tâm điểm (centroid) trên dữ liệu đã reproject -> sẽ chính xác hơn
        gdf_edges_proj['centroid'] = gdf_edges_proj.geometry.centroid
        
        # Tạo GeoDataFrame chỉ chứa centroid để thực hiện spatial join
        edges_centroids_gdf_proj = gpd.GeoDataFrame(
            gdf_edges_proj.drop(columns=['geometry']), # Bỏ cột geometry cũ (LineString)
            geometry=gdf_edges_proj['centroid'],      # Sử dụng cột centroid mới làm geometry
            crs=PROJECTED_CRS
        )

        # 2. Thực hiện Spatial Join
        # Chỉ lấy các cột cần thiết từ taxi_zones_gdf_proj
        zones_to_join_proj = taxi_zones_gdf_proj[['LocationID', 'geometry']]
        print("Đang thực hiện spatial join giữa các cạnh và khu vực taxi...")
        # 'predicate' được ưu tiên hơn 'op' trong các phiên bản GeoPandas mới
        edges_with_zones_info = gpd.sjoin(
            edges_centroids_gdf_proj, 
            zones_to_join_proj, 
            how='left', 
            predicate='within' 
        )
        # `edges_with_zones_info` sẽ có index là (u,v,key) từ `gdf_edges_proj` (và `gdf_edges`)
        # và có thêm cột 'LocationID' và 'index_right' (từ taxi_zones_gdf_proj)
        print("Spatial join hoàn tất.")
        # print("5 dòng đầu của edges_with_zones_info (sau sjoin):")
        # print(edges_with_zones_info[['LocationID', 'highway']].head())


    except Exception as e_spatial_ops:
        print(f"Lỗi nghiêm trọng trong quá trình chuẩn bị không gian hoặc spatial join: {e_spatial_ops}")
        print("Không thể tiếp tục gán travel_time dựa trên model.")
        # Fallback: Gán travel_time dựa trên tốc độ fallback chung cho toàn bộ đồ thị
        fallback_speed_overall_mps = get_fallback_speed_mps(target_hour, fallback_median_speed_by_hour)
        print(f"Sử dụng fallback speed chung: {fallback_speed_overall_mps:.2f} m/s cho toàn bộ đồ thị.")
        for u, v, data_edge_G in G.edges(data=True):
            length_m = data_edge_G.get('length')
            if length_m is not None:
                data_edge_G['travel_time'] = float(length_m) / fallback_speed_overall_mps
            else:
                data_edge_G['travel_time'] = float('inf')
        return G


    # 3. Lặp qua các cạnh của G để tính toán và gán travel_time
    num_edges_fallback_speed = 0
    num_edges_ml_speed = 0
    fallback_speed_mps_value = get_fallback_speed_mps(target_hour, fallback_median_speed_by_hour)

    for u, v, key, data_edge_G in G.edges(keys=True, data=True):
        length_m = data_edge_G.get('length')
        base_speed_mph_for_edge = None # Tốc độ cơ sở (mph) cho cạnh này, sẽ được dự đoán hoặc fallback
        
        try:
            location_id_series = edges_with_zones_info.loc[(u,v,key), 'LocationID'] if (u,v,key) in edges_with_zones_info.index else pd.Series([pd.NA])
            
            # sjoin có thể trả về nhiều dòng nếu centroid nằm trên biên giới của nhiều zone (hiếm)
            # Hoặc nếu một cạnh (u,v,key) không có trong edges_with_zones_info.index (do lỗi join)
            # Lấy giá trị LocationID đầu tiên nếu là Series, hoặc giá trị đơn nếu không phải
            location_id = None
            if not isinstance(location_id_series, pd.Series): # Nếu chỉ là giá trị đơn
                 location_id = location_id_series
            elif not location_id_series.empty:
                 location_id = location_id_series.iloc[0]


            if pd.notna(location_id) and location_id is not None:
                location_id = int(location_id)
                X_pred_df = pd.DataFrame(
                    [[location_id, target_hour, target_day_numeric]],
                    columns=['PULocationID', 'pickup_hour', 'pickup_day_of_week']
                )
                X_pred_processed = ml_preprocessor.transform(X_pred_df)
                predicted_speed_mph = ml_model.predict(X_pred_processed)[0]
                base_speed_mph_for_edge = predicted_speed_mph
                num_edges_ml_speed += 1
            else: 
                base_speed_mph_for_edge = fallback_median_speed_by_hour.loc[target_hour] if target_hour in fallback_median_speed_by_hour.index else fallback_median_speed_by_hour.median()
                if pd.isna(base_speed_mph_for_edge): base_speed_mph_for_edge = 10.0
                num_edges_fallback_speed += 1

            base_speed_mps_for_edge_safe = max(base_speed_mph_for_edge * MPH_TO_MPS, 0.1 * MPH_TO_MPS)

            effective_speed_mps = base_speed_mps_for_edge_safe
            highway_attr = data_edge_G.get('highway', 'default')
            if isinstance(highway_attr, list):
                highway_type = highway_attr[0] if highway_attr else 'default'
            else:
                highway_type = highway_attr
            
            modifier = ROAD_TYPE_SPEED_MODIFIERS.get(highway_type, ROAD_TYPE_SPEED_MODIFIERS['default'])
            speed_after_modifier_mps = base_speed_mps_for_edge_safe * modifier
            
            maxspeed_str = data_edge_G.get('maxspeed')
            parsed_max_speed_mph = parse_maxspeed(maxspeed_str)

            if parsed_max_speed_mph is not None and parsed_max_speed_mph > 0:
                max_speed_mps_from_osm = parsed_max_speed_mph * MPH_TO_MPS
                effective_speed_mps = min(max_speed_mps_from_osm, speed_after_modifier_mps)
            else:
                effective_speed_mps = speed_after_modifier_mps
            
            if effective_speed_mps <= 0:
                effective_speed_mps = 0.1 * MPH_TO_MPS

            if length_m is not None:
                length_m = float(length_m)
                data_edge_G['travel_time'] = length_m / effective_speed_mps
            else:
                data_edge_G['travel_time'] = float('inf')

        except Exception as e_edge_loop:
            # print(f"Lỗi khi xử lý cạnh ({u},{v},{key}): {e_edge_loop}. Gán travel_time là vô cực.")
            data_edge_G['travel_time'] = float('inf')
            
    print(f"Hoàn tất cập nhật 'travel_time'. Số cạnh dùng ML speed: {num_edges_ml_speed}, dùng fallback speed: {num_edges_fallback_speed}")
    return G

# --- Hàm calculate_eta_for_route giữ nguyên ---
def calculate_eta_for_route(G_with_times, origin_node, destination_node):
    # ... (Nội dung hàm giữ nguyên như lần sửa trước) ...
    if G_with_times is None or origin_node is None or destination_node is None:
        print("Lỗi: Thiếu thông tin đồ thị hoặc điểm đầu/cuối để tính ETA.")
        return None, None
    if not G_with_times.has_node(origin_node) or not G_with_times.has_node(destination_node):
        print(f"LỖI: Nút xuất phát {origin_node} hoặc nút đích {destination_node} không tồn tại trong đồ thị.")
        return None, None
    try:
        print(f"Đang tìm lộ trình từ {origin_node} đến {destination_node} (sử dụng 'travel_time' làm trọng số)...")
        route = ox.shortest_path(G_with_times, origin_node, destination_node, weight="travel_time")
        if route:
            print("Tìm thấy lộ trình!")
            route_edges_gdf = ox.routing.route_to_gdf(G_with_times, route)
            if 'travel_time' in route_edges_gdf.columns:
                if route_edges_gdf['travel_time'].isnull().any() or np.isinf(route_edges_gdf['travel_time']).any():
                    print("CẢNH BÁO: Một số đoạn đường trong lộ trình có travel_time không hợp lệ (NaN hoặc Inf).")
                total_travel_time_seconds = route_edges_gdf['travel_time'].sum()
                if np.isinf(total_travel_time_seconds):
                    print("ETA không thể tính được do có đoạn đường không thể đi qua (travel_time = vô cực).")
                    eta_minutes = float('inf')
                elif pd.isna(total_travel_time_seconds):
                    print("ETA không thể tính được do thiếu dữ liệu travel_time (NaN).")
                    eta_minutes = float('nan')
                else:
                    eta_minutes = total_travel_time_seconds / 60
                print(f"Thời gian di chuyển dự kiến (ETA): {total_travel_time_seconds:.2f} giây ({eta_minutes:.2f} phút)")
                return route, eta_minutes
            else:
                print("LỖI: Cột 'travel_time' không có trong GeoDataFrame của lộ trình (route_edges_gdf).")
                print("Các cột có trong route_edges_gdf:", route_edges_gdf.columns)
                return route, None 
        else:
            print("Không tìm thấy lộ trình giữa hai điểm đã chọn.")
            return None, None
    except Exception as e:
        print(f"Lỗi khi tìm lộ trình hoặc tính ETA: {e}")
        return None, None

# --- Phần if __name__ == '__main__': không thay đổi nhiều, chỉ là cách gọi test ---