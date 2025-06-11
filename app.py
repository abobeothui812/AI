# app.py
import streamlit as st

from data_loader import load_taxi_trip_data

# --- DÒNG NÀY PHẢI LÀ LỆNH STREAMLIT ĐẦU TIÊN ---
st.set_page_config(page_title="Ứng dụng Dự đoán ETA Manhattan", layout="wide", initial_sidebar_state="collapsed")
# -------------------------------------------------

import pandas as pd
import numpy as np
import osmnx as ox
import joblib
import folium 
from streamlit_folium import st_folium 

try:
    from config import DEFAULT_TARGET_HOUR, DEFAULT_TARGET_DAY_NUMERIC, PLACE_NAME, YELLOW_TAXI_DATA_FILE
    from data_loader import load_road_network, load_taxi_zones 
    from data_processor import (
        filter_taxi_zones_by_borough,
        initial_trip_data_cleaning,
        filter_trips_by_location_ids,
        calculate_median_speed_by_time,
        create_ml_training_data
    )
    from routing_utils import (
        add_travel_times_to_graph,
        calculate_eta_for_route
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:
    st.error(f"Lỗi import module cục bộ: {e}. Đảm bảo bạn đang chạy 'streamlit run app.py' từ thư mục gốc của dự án và tất cả các file .py cần thiết đều có mặt.")
    st.stop()


# --- Các hàm được Cache của Streamlit ---

@st.cache_resource(show_spinner="Đang tải dữ liệu bản đồ và mô hình ML...")
def load_core_data_cached():
    print("Thực thi: load_core_data_cached()")
    g_manhattan, taxi_zones, model, preprocessor = None, None, None, None
    error_messages = []
    try:
        g_manhattan = load_road_network(place_name=PLACE_NAME)
        if g_manhattan is None: error_messages.append("Lỗi: Không tải được bản đồ Manhattan.")
    except Exception as e: error_messages.append(f"Lỗi nghiêm trọng khi tải bản đồ Manhattan: {e}")
    try:
        taxi_zones = load_taxi_zones()
        if taxi_zones is None: error_messages.append("Lỗi: Không tải được dữ liệu Taxi Zones.")
    except Exception as e: error_messages.append(f"Lỗi nghiêm trọng khi tải Taxi Zones: {e}")
    if not error_messages:
        try:
            model = joblib.load('trained_rf_model.joblib')
            preprocessor = joblib.load('data_preprocessor.joblib')
        except FileNotFoundError:
            error_messages.append("LƯU Ý: Không tìm thấy file model/preprocessor. Sẽ thử huấn luyện lại.")
            model, preprocessor = None, None
        except Exception as e: 
            error_messages.append(f"Lỗi khi tải model/preprocessor: {e}")
            model, preprocessor = None, None
    return g_manhattan, taxi_zones, model, preprocessor, error_messages

@st.cache_data(show_spinner="Đang tính toán tốc độ fallback...")
def get_fallback_speeds_cached(taxi_trip_file_path_param, manhattan_loc_ids_list_param):
    print("Thực thi: get_fallback_speeds_cached()")
    fallback_speeds, error_msg = pd.Series([10.0]*24, index=range(24)), "Sử dụng tốc độ fallback mặc định ban đầu."
    if not manhattan_loc_ids_list_param:
        error_msg = "Không có LocationID của Manhattan cho fallback speeds. " + error_msg
        return fallback_speeds, error_msg
    try:
        import os
        if not os.path.exists(taxi_trip_file_path_param):
            raise FileNotFoundError(f"File dữ liệu taxi '{taxi_trip_file_path_param}' không tồn tại.")
        raw_taxi_trips = pd.read_parquet(taxi_trip_file_path_param)
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            raw_taxi_trips[col] = pd.to_datetime(raw_taxi_trips[col])
        cleaned_taxi_trips = initial_trip_data_cleaning(raw_taxi_trips)
        manhattan_trips = pd.DataFrame()
        if cleaned_taxi_trips is not None and not cleaned_taxi_trips.empty:
            manhattan_trips = filter_trips_by_location_ids(cleaned_taxi_trips, manhattan_loc_ids_list_param)
        if not manhattan_trips.empty:
            median_hr, _ = calculate_median_speed_by_time(manhattan_trips)
            if not median_hr.empty: fallback_speeds, error_msg = median_hr, None
            else: error_msg = "Không tính được median_speed_by_hour từ dữ liệu. " + error_msg
        else: error_msg = "Không có chuyến đi Manhattan nào để tính fallback speed. " + error_msg
    except FileNotFoundError as e_fnf: error_msg = str(e_fnf)
    except Exception as e: error_msg = f"Lỗi khi tính fallback speeds: {e}. " + error_msg
    return fallback_speeds, error_msg

# --- Khởi tạo Session State ---
default_map_center = [40.7679, -73.9822]
default_map_zoom = 12
for key, default_val in [
    ('origin_coords', None), 
    ('destination_coords', None), 
    ('route_nodes', None),
    ('map_center', default_map_center), 
    ('map_zoom', default_map_zoom),
    ('origin_input', "Times Square, New York, NY"), 
    ('destination_input', "Wall Street, New York, NY"),
    ('hour_input', DEFAULT_TARGET_HOUR),
    ('day_input', DEFAULT_TARGET_DAY_NUMERIC),
    ('click_mode', 'Điểm xuất phát')
]:
    if key not in st.session_state:
        st.session_state[key] = default_val

# --- Tải dữ liệu khởi tạo ---
G_manhattan, taxi_zones_gdf, ml_model, ml_preprocessor, initial_errors = load_core_data_cached()
if initial_errors:
    for err in initial_errors:
        if "LƯU Ý" in err: st.info(err)
        else: st.error(err)

manhattan_zones_gdf_filtered = None
if taxi_zones_gdf is not None:
    manhattan_zones_gdf_filtered, manhattan_location_ids_for_fallback = filter_taxi_zones_by_borough(taxi_zones_gdf)
else:
    manhattan_location_ids_for_fallback = []

fallback_median_speed_by_hour, fallback_errors = get_fallback_speeds_cached(
    YELLOW_TAXI_DATA_FILE, manhattan_location_ids_for_fallback
)
if fallback_errors: st.warning(fallback_errors)

# --- Logic Huấn luyện lại Model ---
if ml_model is None and G_manhattan is not None and taxi_zones_gdf is not None and not any("Lỗi nghiêm trọng" in str(err) for err in initial_errors if err is not None):
    st.info("Mô hình ML chưa được tải. Đang thử huấn luyện lại...")
    with st.spinner("Đang huấn luyện lại mô hình ML... (có thể mất vài phút)"):
        # ... (Code huấn luyện lại giữ nguyên) ...
        try:
            raw_taxi_trips_for_retrain = load_taxi_trip_data(YELLOW_TAXI_DATA_FILE)
            if raw_taxi_trips_for_retrain is None: st.error("Không thể tải dữ liệu taxi để huấn luyện lại mô hình.")
            else:
                cleaned_taxi_for_retrain = initial_trip_data_cleaning(raw_taxi_trips_for_retrain)
                manhattan_trips_for_retrain = filter_trips_by_location_ids(cleaned_taxi_for_retrain, manhattan_location_ids_for_fallback)
                if manhattan_trips_for_retrain.empty: st.error("Không có dữ liệu Manhattan để huấn luyện lại mô hình.")
                else:
                    ml_training_df_retrain = create_ml_training_data(manhattan_trips_for_retrain)
                    if ml_training_df_retrain.empty: st.error("Không thể tạo dữ liệu huấn luyện ML cho việc huấn luyện lại.")
                    else:
                        X_retrain = ml_training_df_retrain[['PULocationID', 'pickup_hour', 'pickup_day_of_week']]
                        y_retrain = ml_training_df_retrain['target_median_speed_mph']
                        preprocessor_retrain = ColumnTransformer(
                            transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['PULocationID'])],
                            remainder='passthrough')
                        X_processed_retrain = preprocessor_retrain.fit_transform(X_retrain)
                        ml_model_retrain = RandomForestRegressor(
                            n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, 
                            max_depth=20, min_samples_split=5, min_samples_leaf=2)
                        ml_model_retrain.fit(X_processed_retrain, y_retrain)
                        joblib.dump(ml_model_retrain, 'trained_rf_model.joblib')
                        joblib.dump(preprocessor_retrain, 'data_preprocessor.joblib')
                        st.success("Đã huấn luyện lại và lưu mô hình, preprocessor thành công! Vui lòng làm mới trang để sử dụng.")
                        ml_model, ml_preprocessor = ml_model_retrain, preprocessor_retrain
        except Exception as e_retrain: st.error(f"Lỗi trong quá trình huấn luyện lại mô hình: {e_retrain}")


# --- Hàm tạo và cập nhật bản đồ ---
def render_map(graph_map, origin_coords, dest_coords, route_nodes, map_bounds):
    map_center = st.session_state.map_center
    map_zoom = st.session_state.map_zoom
    if isinstance(map_center, dict): map_location = [map_center.get('lat'), map_center.get('lng')]
    else: map_location = list(map_center)
    m = folium.Map(location=map_location, zoom_start=map_zoom, tiles="CartoDB positron", max_bounds=map_bounds, min_zoom=11)
    if origin_coords: folium.Marker(location=origin_coords, popup="Điểm xuất phát", icon=folium.Icon(color='green', icon='play')).add_to(m)
    if dest_coords: folium.Marker(location=dest_coords, popup="Điểm đến", icon=folium.Icon(color='red', icon='stop')).add_to(m)
    if route_nodes and graph_map and len(route_nodes) > 1:
        try:
            route_gdf = ox.routing.route_to_gdf(graph_map, route_nodes)
            folium.GeoJson(route_gdf, style_function=lambda x: {'color': "#007bff", 'weight': 5, 'opacity': 0.7}).add_to(m)
            if not route_gdf.empty:
                bounds = route_gdf.total_bounds
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        except Exception as e: st.warning(f"Lỗi khi vẽ lộ trình: {e}.")
    return m

# --- BỐ CỤC CHÍNH CỦA ỨNG DỤNG ---
st.title("Ứng dụng Dự đoán ETA ở Manhattan")

control_col, map_col = st.columns([1, 2]) 

with control_col:
    st.header("Nhập thông tin lộ trình")
    
    if all(obj is not None for obj in [G_manhattan, ml_model, ml_preprocessor, fallback_median_speed_by_hour, taxi_zones_gdf]):
        
        st.radio("Chế độ Chọn trên Bản đồ:", ('Điểm xuất phát', 'Điểm đến'), key='click_mode', horizontal=True)
        st.caption("Sau khi chọn chế độ, hãy click vào một điểm trên bản đồ để đặt marker.")

        # --- SỬA LỖI Ở ĐÂY: Thêm callback on_change ---
        def geocode_and_update(address_key, coords_key):
            """Callback để geocode địa chỉ từ text input và cập nhật session_state."""
            address = st.session_state[address_key]
            if address:
                try:
                    lat, lon = ox.geocode(address)
                    st.session_state[coords_key] = (lat, lon)
                    st.session_state.route_nodes = None # Xóa route cũ khi địa chỉ thay đổi
                    st.toast(f"Đã tìm thấy: {address}", icon="📍")
                except Exception:
                    st.session_state[coords_key] = None
                    st.toast(f"Không tìm thấy địa chỉ: '{address}'", icon="⚠️")
            else:
                st.session_state[coords_key] = None # Xóa marker nếu input rỗng

        st.text_input("📍 Điểm xuất phát", key="origin_input", on_change=geocode_and_update, args=("origin_input", "origin_coords"))
        st.text_input("🏁 Điểm đến", key="destination_input", on_change=geocode_and_update, args=("destination_input", "destination_coords"))
        
        st.write("🕒 Thời gian khởi hành")
        
        days_options = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Giờ (0-23)", min_value=0, max_value=23, key="hour_input", step=1)
        with c2:
            st.selectbox("Ngày", options=list(days_options.keys()), format_func=lambda x: days_options[x], key="day_input")

        if st.button("TÍNH TOÁN ETA", use_container_width=True, type="primary"):
            if not st.session_state.origin_input or not st.session_state.destination_input:
                st.warning("Vui lòng nhập hoặc chọn cả điểm xuất phát và điểm đến.")
            elif st.session_state.origin_coords is None or st.session_state.destination_coords is None:
                st.warning("Vui lòng đảm bảo cả hai địa chỉ đã được geocode thành công.")
            else:
                with st.spinner("Đang tính toán..."):
                    origin_node = ox.nearest_nodes(G_manhattan, X=st.session_state.origin_coords[1], Y=st.session_state.origin_coords[0])
                    dest_node = ox.nearest_nodes(G_manhattan, X=st.session_state.destination_coords[1], Y=st.session_state.destination_coords[0])
                    
                    if origin_node and dest_node:
                        G_for_eta = G_manhattan.copy()
                        add_travel_times_to_graph(
                            G_for_eta, st.session_state.hour_input, st.session_state.day_input, ml_model,
                            ml_preprocessor, taxi_zones_gdf, fallback_median_speed_by_hour
                        )
                        route, eta_minutes = calculate_eta_for_route(G_for_eta, origin_node, dest_node)
                        st.session_state.route_nodes = route

                        if route and eta_minutes is not None and not pd.isna(eta_minutes) and not np.isinf(eta_minutes):
                            route_gdf = ox.routing.route_to_gdf(G_for_eta, route)
                            total_distance_meters = route_gdf['length'].sum()
                            total_distance_km = total_distance_meters / 1000
                            
                            st.session_state.last_eta = eta_minutes
                            st.session_state.last_distance = total_distance_km
                        else:
                            st.session_state.route_nodes = None
                            st.session_state.last_eta = None
                            st.session_state.last_distance = None
                            if route is None: st.error("Không tìm thấy lộ trình.")
                            else: st.error(f"Không thể tính ETA (ETA = {eta_minutes}).")
                    else: st.error("Không thể tìm thấy nút mạng lưới đường gần nhất.")
                st.rerun()

        # Khu vực hiển thị kết quả
        if 'last_eta' in st.session_state and st.session_state.last_eta is not None:
            st.subheader("Kết quả Lộ trình")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("ETA Dự kiến", f"{st.session_state.last_eta:.1f} phút")
            with res_col2:
                st.metric("Tổng Quãng đường", f"{st.session_state.last_distance:.2f} km")
    else:
        st.warning("Vui lòng đợi dữ liệu và mô hình được tải xong hoặc khắc phục lỗi hiển thị ở trên.")

with map_col:
    st.header("Bản đồ Lộ trình")
    manhattan_bounds = None
    if manhattan_zones_gdf_filtered is not None and not manhattan_zones_gdf_filtered.empty:
        bounds_array = manhattan_zones_gdf_filtered.total_bounds
        manhattan_bounds = [[bounds_array[1], bounds_array[0]], [bounds_array[3], bounds_array[2]]]

    if G_manhattan: 
        folium_map = render_map(
            G_manhattan, 
            st.session_state.origin_coords, 
            st.session_state.destination_coords,
            st.session_state.route_nodes,
            manhattan_bounds
        )
        map_output = st_folium(folium_map, key="map", width=700, height=700, returned_objects=["last_clicked", "center", "zoom"])

        if map_output and map_output.get("last_clicked"):
            lat, lon = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]
            
            if st.session_state.click_mode == 'Điểm xuất phát':
                st.session_state.origin_coords = (lat, lon)
                coord_key_to_update = "origin_input"
            else:
                st.session_state.destination_coords = (lat, lon)
                coord_key_to_update = "destination_input"
            
            try:
                address = ox.reverse_geocode(lat, lon)
                st.session_state[coord_key_to_update] = address
            except Exception as e:
                st.warning(f"Không thể tìm thấy địa chỉ cho tọa độ đã chọn: {e}")
                # Giữ lại giá trị input cũ nếu reverse geocode thất bại
            
            st.session_state.route_nodes = None
            st.rerun()
    else:
        st.warning("Đang chờ tải dữ liệu bản đồ...")