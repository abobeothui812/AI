# data_loader.py
import pandas as pd
import geopandas as gpd
import osmnx as ox
from config import PLACE_NAME, TAXI_ZONES_SHAPEFILE_PATH, YELLOW_TAXI_DATA_FILE

def load_road_network(place_name=PLACE_NAME, network_type="drive"):
    """Tải dữ liệu mạng lưới đường từ OpenStreetMap."""
    print(f"Đang tải dữ liệu mạng lưới đường cho {place_name}...")
    try:
        G = ox.graph_from_place(place_name, network_type=network_type, retain_all=True)
        print("Tải dữ liệu mạng lưới đường hoàn tất!")
        return G
    except Exception as e:
        print(f"Lỗi khi tải mạng lưới đường: {e}")
        return None

def load_taxi_zones(shapefile_path=TAXI_ZONES_SHAPEFILE_PATH):
    """Tải dữ liệu shapefile của các Khu vực Taxi."""
    print(f"Đang đọc file Shapefile Khu vực Taxi từ: {shapefile_path}...")
    try:
        taxi_zones_gdf = gpd.read_file(shapefile_path)
        print("Đọc file Shapefile Khu vực Taxi thành công!")
        if 'LocationID' in taxi_zones_gdf.columns:
            taxi_zones_gdf['LocationID'] = taxi_zones_gdf['LocationID'].astype(int)
        return taxi_zones_gdf
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file Shapefile Khu vực Taxi tại {shapefile_path}.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc Shapefile Khu vực Taxi: {e}")
        return None

def load_taxi_trip_data(parquet_file_path=YELLOW_TAXI_DATA_FILE):
    """Tải dữ liệu chuyến đi taxi từ file Parquet."""
    print(f"Đang đọc file dữ liệu taxi: {parquet_file_path}...")
    try:
        df_taxi = pd.read_parquet(parquet_file_path)
        print("Đọc file dữ liệu taxi hoàn tất!")
        # Chuyển đổi cột thời gian cơ bản
        df_taxi['tpep_pickup_datetime'] = pd.to_datetime(df_taxi['tpep_pickup_datetime'])
        df_taxi['tpep_dropoff_datetime'] = pd.to_datetime(df_taxi['tpep_dropoff_datetime'])
        return df_taxi
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file taxi {parquet_file_path}.")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu taxi: {e}")
        return None

if __name__ == '__main__':
    # Chạy thử các hàm load (chỉ khi chạy trực tiếp file này)
    G_test = load_road_network()
    if G_test:
        print(f"Đồ thị có {len(G_test.nodes)} nút và {len(G_test.edges)} cạnh.")

    taxi_zones_test = load_taxi_zones()
    if taxi_zones_test is not None:
        print(taxi_zones_test.head())

    taxi_data_test = load_taxi_trip_data()
    if taxi_data_test is not None:
        print(taxi_data_test.head())