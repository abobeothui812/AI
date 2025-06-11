# data_processor.py
import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox

from config import (
    TARGET_BOROUGH, MIN_TRIP_DURATION_MINUTES, MAX_TRIP_DURATION_MINUTES,
    MIN_TRIP_DISTANCE_MILES, MIN_AVG_SPEED_MPH, MAX_AVG_SPEED_MPH
)

def filter_taxi_zones_by_borough(taxi_zones_gdf, borough_name=TARGET_BOROUGH):
    if taxi_zones_gdf is None: return None, []
    borough_zones = taxi_zones_gdf[taxi_zones_gdf['borough'] == borough_name]
    borough_location_ids = borough_zones['LocationID'].tolist()
    #print(f"Số lượng khu vực taxi ở {borough_name}: {len(borough_zones)}")
    return borough_zones, borough_location_ids

def initial_trip_data_cleaning(df_taxi):
    if df_taxi is None: return None
    df_cleaned = df_taxi.copy()
    df_cleaned['trip_duration_seconds'] = (df_cleaned['tpep_dropoff_datetime'] - df_cleaned['tpep_pickup_datetime']).dt.total_seconds()
    df_cleaned['trip_duration_minutes'] = df_cleaned['trip_duration_seconds'] / 60
    df_filtered = df_cleaned[
        (df_cleaned['trip_duration_minutes'] >= MIN_TRIP_DURATION_MINUTES) &
        (df_cleaned['trip_duration_minutes'] <= MAX_TRIP_DURATION_MINUTES) &
        (df_cleaned['trip_distance'] >= MIN_TRIP_DISTANCE_MILES)
    ].copy()
    valid_duration_mask = df_filtered['trip_duration_seconds'] > 0
    df_filtered.loc[valid_duration_mask, 'average_speed_mph'] = \
        df_filtered.loc[valid_duration_mask, 'trip_distance'] / \
        (df_filtered.loc[valid_duration_mask, 'trip_duration_seconds'] / 3600)
    if 'average_speed_mph' in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered['average_speed_mph'] >= MIN_AVG_SPEED_MPH) &
            (df_filtered['average_speed_mph'] <= MAX_AVG_SPEED_MPH)
        ]
    else:
        if not df_filtered.empty: df_filtered['average_speed_mph'] = pd.NA
    print(f"Số chuyến đi sau khi lọc cơ bản và lọc tốc độ: {len(df_filtered)}")
    return df_filtered

def filter_trips_by_location_ids(df_taxi_filtered, location_ids):
    if df_taxi_filtered is None or not location_ids: return pd.DataFrame()
    df_filtered = df_taxi_filtered.copy()
    if 'PULocationID' in df_filtered.columns: df_filtered['PULocationID'] = df_filtered['PULocationID'].astype(int)
    if 'DOLocationID' in df_filtered.columns: df_filtered['DOLocationID'] = df_filtered['DOLocationID'].astype(int)
    df_borough_trips = df_filtered[
        df_filtered['PULocationID'].isin(location_ids) &
        df_filtered['DOLocationID'].isin(location_ids)
    ]
    print(f"Số chuyến đi hoàn toàn trong các khu vực đã chọn: {len(df_borough_trips)}")
    return df_borough_trips

def calculate_median_speed_by_time(df_borough_trips):
    if df_borough_trips is None or df_borough_trips.empty or 'average_speed_mph' not in df_borough_trips.columns:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    df_analysis = df_borough_trips.copy()
    df_analysis['pickup_hour'] = pd.to_datetime(df_analysis['tpep_pickup_datetime']).dt.hour
    df_analysis['pickup_day_of_week'] = pd.to_datetime(df_analysis['tpep_pickup_datetime']).dt.dayofweek
    median_speed_by_hour = df_analysis.groupby('pickup_hour')['average_speed_mph'].median()
    median_speed_by_day_of_week = df_analysis.groupby('pickup_day_of_week')['average_speed_mph'].median()
    days = ['Thứ Hai', 'Thứ Ba', 'Thứ Tư', 'Thứ Năm', 'Thứ Sáu', 'Thứ Bảy', 'Chủ Nhật']
    median_speed_by_day_of_week.index = median_speed_by_day_of_week.index.map(lambda x: days[x] if x < len(days) else x)
    return median_speed_by_hour, median_speed_by_day_of_week

def calculate_spatial_features(taxi_zones_gdf, G):
    """
    Tính toán các đặc trưng không gian cho mỗi Khu vực Taxi.
    Ví dụ: Mật độ đường sá.
    """
    print("Đang tính toán các đặc trưng không gian cho các khu vực taxi...")
    if taxi_zones_gdf is None or G is None:
        return pd.DataFrame()

    # 1. Reproject sang CRS phẳng để tính toán chính xác (UTM Zone 18N cho NYC)
    projected_crs = "EPSG:32618"
    zones_proj = taxi_zones_gdf.to_crs(projected_crs)
    _, edges_proj = ox.graph_to_gdfs(ox.project_graph(G, to_crs=projected_crs))

    # 2. Tính diện tích cho mỗi zone (mét vuông)
    zones_proj['zone_area_sqm'] = zones_proj.geometry.area

    # 3. Spatial join để tìm các đoạn đường trong mỗi zone
    edges_in_zones = gpd.sjoin(edges_proj, zones_proj, how='inner', predicate='intersects')

    # 4. Tính tổng chiều dài đường cho mỗi zone
    road_length_per_zone = edges_in_zones.groupby('LocationID')['length'].sum()

    # 5. Hợp nhất (merge) dữ liệu và tính toán đặc trưng
    spatial_features_df = zones_proj[['LocationID', 'zone_area_sqm']].copy()
    spatial_features_df = spatial_features_df.merge(road_length_per_zone.rename('total_road_length_m'), on='LocationID', how='left')
    spatial_features_df['total_road_length_m'].fillna(0, inplace=True)

    # Tính mật độ đường sá (m / m^2)
    spatial_features_df['road_density'] = spatial_features_df['total_road_length_m'] / spatial_features_df['zone_area_sqm']
    
    print("Tính toán đặc trưng không gian hoàn tất.")
    # Chỉ trả về các cột cần thiết
    return spatial_features_df[['LocationID', 'road_density']]


def create_ml_training_data(df_borough_trips):
    """
    Tạo dữ liệu huấn luyện cho mô hình ML, bao gồm các đặc trưng thời gian tuần hoàn.
    """
    if df_borough_trips is None or df_borough_trips.empty or 'average_speed_mph' not in df_borough_trips.columns:
        return pd.DataFrame()

    df_ml = df_borough_trips.copy()

    # Trích xuất các thành phần thời gian cơ bản
    df_ml['pickup_hour'] = pd.to_datetime(df_ml['tpep_pickup_datetime']).dt.hour
    df_ml['pickup_day_of_week'] = pd.to_datetime(df_ml['tpep_pickup_datetime']).dt.dayofweek

    # --- TẠO ĐẶC TRƯNG THỜI GIAN TUẦN HOÀN (CYCLICAL FEATURES) ---
    df_ml['hour_sin'] = np.sin(2 * np.pi * df_ml['pickup_hour']/24.0)
    df_ml['hour_cos'] = np.cos(2 * np.pi * df_ml['pickup_hour']/24.0)
    df_ml['day_sin'] = np.sin(2 * np.pi * df_ml['pickup_day_of_week']/7.0)
    df_ml['day_cos'] = np.cos(2 * np.pi * df_ml['pickup_day_of_week']/7.0)

    # Nhóm dữ liệu để tính tốc độ trung vị (target)
    # Bao gồm cả các đặc trưng mới trong groupby để giữ chúng lại
    grouping_cols = [
        'PULocationID', 'pickup_hour', 'pickup_day_of_week',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    df_zone_speed = df_ml.groupby(grouping_cols)['average_speed_mph'].median().reset_index()
    
    df_zone_speed.rename(columns={'average_speed_mph': 'target_median_speed_mph'}, inplace=True)
    df_zone_speed.dropna(subset=['target_median_speed_mph'], inplace=True)
    
    print(f"Đã tạo được {len(df_zone_speed)} mẫu dữ liệu huấn luyện ML với các đặc trưng mới.")
    
    # Loại bỏ các cột thời gian gốc vì đã có biểu diễn sin/cos
    # df_zone_speed = df_zone_speed.drop(columns=['pickup_hour', 'pickup_day_of_week'])
    
    return df_zone_speed

