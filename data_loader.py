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