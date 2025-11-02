import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud("./pc_color_filtered.pcd")
print(pcd)
print(np.asarray(pcd.points))  # coordinate XYZ
print(np.asarray(pcd.colors))  # colori RGB (normalizzati tra 0 e 1)
o3d.visualization.draw_geometries([pcd])

# labels = np.asarray(pcd.colors)[:, 0] * 255  # esempio se label codificata in colore
# unique_labels = np.unique(labels)

# colors = plt.get_cmap("tab20")(labels / max(labels))
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])