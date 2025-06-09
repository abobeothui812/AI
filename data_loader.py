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