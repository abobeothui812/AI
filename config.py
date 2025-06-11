# config.py

# --- Cấu hình Địa điểm và Dữ liệu ---
PLACE_NAME = "Manhattan, New York City, New York, USA"
TAXI_ZONES_SHAPEFILE_PATH = "data/nyc_taxi_zones/taxi_zones.shp" # Điều chỉnh nếu cần
# Đặt tên file dữ liệu taxi của bạn ở đây
YELLOW_TAXI_DATA_FILE = "data/yellow_tripdata_2025-01.parquet" # Điều chỉnh nếu cần

# --- Cấu hình Phân tích ---
TARGET_BOROUGH = "Manhattan"
DEFAULT_TARGET_HOUR = 10  # Giờ mặc định để tính ETA (ví dụ: 10 giờ sáng)
DEFAULT_TARGET_DAY_NUMERIC = 1  # Ngày mặc định (0=Thứ Hai, 1=Thứ Ba, ..., 6=Chủ Nhật)

# --- Cấu hình Xử lý Dữ liệu ---
MIN_TRIP_DURATION_MINUTES = 1
MAX_TRIP_DURATION_MINUTES = 120
MIN_TRIP_DISTANCE_MILES = 0.01 # Lớn hơn 0 một chút
MIN_AVG_SPEED_MPH = 0.5
MAX_AVG_SPEED_MPH = 80 # Giới hạn tốc độ hợp lý cho phân tích

# --- Chuyển đổi đơn vị ---
MPH_TO_MPS = 0.44704 # 1 dặm/giờ = 0.44704 mét/giây

ROAD_TYPE_SPEED_MODIFIERS = {
    'motorway': 1.3,
    'motorway_link': 1.2,
    'trunk': 1.2,
    'trunk_link': 1.1,
    'primary': 1.1,         # Đường chính
    'primary_link': 1.0,
    'secondary': 1.0,       # Đường thứ cấp
    'secondary_link': 0.9,
    'tertiary': 0.9,        # Đường cấp ba
    'tertiary_link': 0.8,
    'residential': 0.7,     # Đường trong khu dân cư
    'unclassified': 0.8,    # Đường không phân loại
    'living_street': 0.5,
    'service': 0.6,         # Đường dịch vụ, đường vào tòa nhà, bãi đỗ xe
    'road': 0.8,            # Loại chung chung, nếu có
    'default': 0.7          # Hệ số mặc định cho các loại đường không có trong danh sách
}


MODEL_PATH = "trained_rf_model.joblib"
PREPROCESSOR_PATH = "data_preprocessor.joblib"