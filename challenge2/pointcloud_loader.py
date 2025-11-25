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

    # Colore della linea
    line_color = np.array([[1, 0, 1]])  # RGB
    line_set.colors = o3d.utility.Vector3dVector(line_color)

    linesets.append(line_set)


# =========================================================
# CREA POINTCLOUD OPEN3D E VISUALIZZA
# =========================================================

def line_to_cylinder(start, end, radius=0.01, color=[1,0,1], resolution=20):
    """
    Crea un cilindro che collega `start` e `end`.
    - start, end: array-like (3,)
    - radius: raggio del cilindro
    - color: lista o array RGB nel range [0,1]
    - resolution: risoluzione del cilindro (numero di lati)
    Restituisce un o3d.geometry.TriangleMesh oppure None se start==end.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    vec = end - start
    length = np.linalg.norm(vec)
    if length == 0.0:
        return None

    # crea cilindro centrato all'origine con asse z e altezza = length
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution, split=4)
    cylinder.compute_vertex_normals()

    # colore
    color = np.asarray(color, dtype=float)
    if color.size == 3:
        color = np.clip(color, 0.0, 1.0)
        cylinder.paint_uniform_color(color.tolist())

    # direzione target (unitario)
    v = vec / length
    z_axis = np.array([0.0, 0.0, 1.0])

    # calcola rotazione: vogliamo ruotare z_axis -> v
    # caso 1: vettori quasi uguali -> nessuna rotazione
    dot = np.dot(z_axis, v)
    eps = 1e-6
    if dot > 1.0 - eps:
        R = np.eye(3)
    elif dot < -1.0 + eps:
        # z and v sono opposti: rotazione di pi attorno a qualsiasi asse ortogonale (es. x)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
    else:
        axis = np.cross(z_axis, v)
        axis_norm = np.linalg.norm(axis)
        axis_unit = axis / axis_norm
        angle = np.arccos(dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_unit * angle)

    # ruota il cilindro attorno all'origine (è centrato sull'origine)
    cylinder.rotate(R, center=np.zeros(3))

    # trasla al midpoint (perché il cilindro è centrato nell'origine)
    midpoint = (start + end) / 2.0
    cylinder.translate(midpoint)

    return cylinder


# converti tutte le linee in cilindri per avere uno spessore maggiore
cylinders = []
for line_set in linesets:
    pts = np.asarray(line_set.points)
    if pts.shape[0] >= 2:
        cyl = line_to_cylinder(pts[0], pts[1], radius=0.02, color=[1,1,0])
        if cyl is not None:
            cylinders.append(cyl)

segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(points)
segmented_pcd.colors = o3d.utility.Vector3dVector(colors)
print("[INFO] Visualizing segmented point cloud...")
# o3d.visualization.draw_geometries([segmented_pcd] + linesets)
o3d.visualization.draw_geometries([segmented_pcd] + cylinders)

# =========================================================
# PCA per ogni ramo (TODO: aggiungere emoji fuoco)
# =========================================================

k = 5  # numero di segmenti che vuoi per ogni ramo
branch_linesets = []

for obj in data["objects"]:
    if obj["classTitle"] != "Branch 1":
        continue

    all_indices = object_to_indices[obj["key"]]
    branch_points = points[all_indices]

    if len(branch_points) < 5:
        continue  # troppo pochi punti per PCA

    # PCA SOLO su questo ramo
    center = branch_points.mean(axis=0)
    pts_centered = branch_points - center

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]  # direzione principale

    # Proiezione dei punti sull’asse
    proj = pts_centered @ axis

    # Suddivisione in k segmenti
    edges = np.linspace(proj.min(), proj.max(), k+1)

    centers = []
    for i in range(k):
        mask = (proj >= edges[i]) & (proj < edges[i+1])
        if np.any(mask):
            centers.append(branch_points[mask].mean(axis=0))

    centers = np.array(centers)
    if len(centers) < 2:
        continue  # non posso creare una polilinea con <2 punti

    # Crea line set per visualizzare i segmenti del ramo
    lines = [[i, i+1] for i in range(len(centers)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in lines])  # rosso

    branch_linesets.append(line_set)

# Visualizzazione
# o3d.visualization.draw_geometries([segmented_pcd] + branch_linesets)
o3d.visualization.draw_geometries(branch_linesets)