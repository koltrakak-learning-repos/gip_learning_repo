import open3d as o3d
import json
import numpy as np
import os
from collections import defaultdict

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
# CARICA ANNOTAZIONI JSON
# =========================================================

print("[INFO] Loading annotations...")
with open(ANN_PATH, "r") as f:
    data = json.load(f)

# =========================================================
# PREPARA ARRAY COLORI
# =========================================================

colors = np.ones_like(points) * 0.5  # default grigio

# Mappa delle classi ai colori
class_colors = {
    "Branch 1": [1, 0, 0],  # rosso
    "Tree": [0, 1, 0],      # verde
}

# =========================================================
# COLORA I PUNTI IN BASE ALLE ANNOTAZIONI
# =========================================================

figures = data.get("figures", [])
# print(f"{len(figures)} rami trovati") # questo è sbagliato le figures non sono solo i rami segmentati

for fig in figures:
    obj = next((o for o in data["objects"] if o["key"] == fig["objectKey"]), None) # generator expression
    if obj is None:
        continue
    base_color = class_colors.get(obj["classTitle"], np.array([0.5, 0.5, 0.5]))
    # Genera una piccola variazione casuale per rendere ogni figura leggermente diversa
    variation = (np.random.rand(3) - 0.5) # tripla di tre valori appartenenti a [-0.5, 0.5] 
    color = np.clip(base_color + variation, 0, 1)  # mantieni valori tra 0 e 1
    
    indices = fig["geometry"]["indices"]
    if len(indices) > 0:
        colors[indices] = color

# print("\n==============================")
# print("   PRIMI 10 PUNTI PER RAMO")
# print("==============================\n")


# Siccome, alcuni oggetti sono spezzati in più figure
# raggruppiamo tutte le figure per objectKey 

# 'defaultdict(list)' è come un dict normale ma se accedi 
# a una chiave che non esiste, viene automaticamente creata 
# con un valore di default (in questo caso lista vuota).
object_to_indices = defaultdict(list) 

for fig in data["figures"]:
    obj_key = fig["objectKey"]
    indices = fig["geometry"]["indices"]
    # estende la lista associata all'objectKey corrente
    object_to_indices[obj_key].extend(indices) 

branch_counter = 0
for obj in data["objects"]:
    if obj["classTitle"] != "Branch 1":
        continue
    # 'key' per objects == 'objectKey' per figures (che sono quelle che ho nella mappa)
    all_indices = object_to_indices[obj["key"]]
    # all_indices = list(set(all_indices))  # eliminiamo duplicati (non penso sia necessario)

    branch_points = points[all_indices]

    branch_counter += 1
    print(f"\n--- Branch {branch_counter} (objectKey={obj["key"]}) ---")
    print("Primi 10 punti:")
    print(branch_points[:10])

print(f"\nTotale rami trovati: {branch_counter}")

# =========================================================
# CREA LE RETTE PER OGNI RAMO (linee tra primo e ultimo punto)
# =========================================================

linesets = []

for obj in data["objects"]:
    if obj["classTitle"] != "Branch 1":
        continue

    all_indices = object_to_indices[obj["key"]]
    if len(all_indices) < 2:
        continue  # non posso fare una retta con <2 punti

    branch_points = points[all_indices]

    start = branch_points[0]
    end   = branch_points[-1]

    # Crea LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([start, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])

    # Colore della linea (giallo)
    line_color = np.array([[1, 1, 0]])  # RGB
    line_set.colors = o3d.utility.Vector3dVector(line_color)

    linesets.append(line_set)


# =========================================================
# CREA POINTCLOUD OPEN3D E VISUALIZZA
# =========================================================

segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(points)
segmented_pcd.colors = o3d.utility.Vector3dVector(colors)
print("[INFO] Visualizing segmented point cloud...")
o3d.visualization.draw_geometries([segmented_pcd] + linesets)
