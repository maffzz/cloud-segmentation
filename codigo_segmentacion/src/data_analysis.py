import open3d as o3d
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


data = o3d.io.read_point_cloud("scan.ply")
points = np.asarray(data.points)

# Número de puntos
print("Número de puntos:", len(points))

# Imagen
#vis = o3d.visualization.Visualizer()
#vis.create_window(window_name="Point Cloud")
#vis.add_geometry(data)
#vis.run()
#vis.capture_screen_image("point_cloud.png")
#vis.destroy_window()

# Atributos
print("Coordenadas:", points)
print("Colores:", np.asarray(data.colors))
print("Normales:", np.asarray(data.normals))

# Densidad
k = 20
tree = o3d.geometry.KDTreeFlann(data)
mean_distances = []
for point in points:
    _, idx, distances = tree.search_knn_vector_3d(point, k + 1)
    distances = np.sqrt(distances[1:])
    mean_distances.append(np.mean(distances))
mean_distances = np.array(mean_distances)

print("\nDENSIDAD")
print("Distancia promedio a vecinos:", np.mean(mean_distances))
print("Distancia mínima:", np.min(mean_distances))
print("Distancia máxima:", np.max(mean_distances))

plt.figure(figsize=(8, 5))
plt.boxplot(mean_distances, orientation="vertical", showfliers=True)
plt.title("Distribución de distancias promedio a 20 vecinos")
plt.ylabel("Distancia promedio")
plt.grid(axis="y", alpha=0.3)
plt.savefig("noise_boxplot2.png", dpi=300, bbox_inches="tight")
plt.close()

# Ruido
filtered, inliers = data.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)
outliers = data.select_by_index(inliers, invert=True)
print("\nRUIDO")
print("Puntos originales:", len(data.points))
print("Puntos atípicos:", len(outliers.points))
print("Porcentaje de posibles puntos ruidosos:",
      len(outliers.points) / len(data.points) * 100, "%")
o3d.io.write_point_cloud("noise.ply", outliers)

# Zonas de baja cobertura
threshold = np.percentile(mean_distances, 95)
sparse_indices = np.where(mean_distances > threshold)[0]
sparse_points = data.select_by_index(sparse_indices)
print("\nZONAS DE BAJA COBERTURA")
print("Puntos en zonas dispersas:", len(sparse_indices))
print("Porcentaje:", len(sparse_indices) / len(points) * 100, "%")
o3d.io.write_point_cloud("sparse_areas.ply", sparse_points)
