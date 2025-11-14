import open3d as o3d
import json
import numpy as np
import os
import glob

# =========================================================
# CONFIG
# =========================================================

PCD_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/pointcloud/pc_color_filtered.pcd"
ANN_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/ann/pc_color_filtered.pcd.json"

# =========================================================
# CARICA POINTCLOUD
# =========================================================

print("[INFO] Loading PCD...")
pcd = o3d.io.read_point_cloud(PCD_PATH)
points = np.asarray(pcd.points)
print(f"[INFO] Loaded {len(points)} points.")

# =========================================================
# CERCA AUTOMATICAMENTE IL FILE JSON DI SUPERVISELY (opzionale)
# =========================================================

if not os.path.exists(ANN_PATH):
    ann_files = glob.glob("ann/*.json")
    if len(ann_files) == 1:
        ANN_PATH = ann_files[0]
        print(f"[INFO] Found annotation file: {ANN_PATH}")
    else:
        raise FileNotFoundError(
            "Nessun file di annotazione trovato in ann/*.json"
        )

# =========================================================
# CARICA ANNOTAZIONI JSON
# =========================================================

print("[INFO] Loading annotations...")
with open(ANN_PATH, "r") as f:
    data = json.load(f)

# =========================================================
# PREPARA ARRAY COLORI
# =========================================================

colors = np.ones_like(points) * 0.5  # default grigio

# Mappa delle classi ai colori (puoi personalizzare)
class_colors = {
    "Branch 1": [1, 0, 0],  # rosso
    "Tree": [0, 1, 0],      # verde
}

# =========================================================
# COLORA I PUNTI IN BASE ALLE ANNOTAZIONI
# =========================================================

figures = data.get("figures", [])
print(f"{len(figures)} rami trovati")

for fig in figures:
    object_key = fig["objectKey"]
    obj = next((o for o in data["objects"] if o["key"] == object_key), None)
    if obj is None:
        continue
    class_name = obj["classTitle"]
    base_color = class_colors.get(class_name, np.array([0.5, 0.5, 0.5]))
    # Genera una piccola variazione casuale per rendere ogni figura leggermente diversa
    variation = (np.random.rand(3) - 0.5) # tripla di tre valori appartenenti a [-0.5, 0.5] 
    color = np.clip(base_color + variation, 0, 1)  # mantieni valori tra 0 e 1
    
    indices = fig["geometry"]["indices"]
    if len(indices) > 0:
        colors[indices] = color

# =========================================================
# CREA POINTCLOUD OPEN3D E VISUALIZZA
# =========================================================

segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(points)
segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

print("[INFO] Visualizing segmented point cloud...")
o3d.visualization.draw_geometries([segmented_pcd])
