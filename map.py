import osmnx as ox
import matplotlib.pyplot as plt

# Tên quận và thành phố
place_name = "Manhattan, New York City, New York, USA"

print(f"Đang tải dữ liệu mạng lưới đường cho {place_name}...")
# Tải dữ liệu mạng lưới đường cho loại 'drive' (đường cho xe ô tô)
# Gdf là viết tắt của Graph from place
G = ox.graph_from_place(place_name, network_type="drive")
print("Tải dữ liệu mạng lưới đường hoàn tất!")

# Chuyển đổi đồ thị sang dạng có thể sử dụng với GeoPandas để lấy thông tin
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

print(f"\nThông tin về mạng lưới đường:")
print(f"Số lượng nút (giao lộ, điểm cuối): {len(gdf_nodes)}")
print(f"Số lượng cạnh (đoạn đường): {len(gdf_edges)}")

# Xem một vài hàng đầu của dữ liệu các đoạn đường (cạnh)
print("\nVí dụ dữ liệu các đoạn đường (cạnh):")
print(gdf_edges.head())

# Vẽ đồ thị mạng lưới đường (có thể mất một chút thời gian tùy thuộc vào kích thước)
print("\nĐang vẽ đồ thị mạng lưới đường...")
fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor="w", node_size=0, edge_color="k", edge_linewidth=0.3)
plt.suptitle(f"Mạng lưới đường bộ của {place_name.split(',')[0]}", y=0.95, fontsize=15)
plt.show()
print("Hoàn tất vẽ đồ thị.")

# Bạn có thể lưu đồ thị lại nếu muốn
# ox.save_graphml(G, filepath="manhattan_drive_network.graphml")