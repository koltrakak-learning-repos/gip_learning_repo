import open3d as o3d
import json
import numpy as np
from collections import defaultdict
from pprint import pprint
import random

import pointcloud_preprocessor as pcpp
import visualization_stuff

PCD_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/pointcloud/pc_color_filtered.pcd"
ANN_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/ann/pc_color_filtered.pcd.json"

pcd = o3d.io.read_point_cloud(PCD_PATH)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
print(f"[INFO] Loaded {len(points)} points.")

print("[INFO] Loading annotations...")
with open(ANN_PATH, "r") as f:
    data = json.load(f)

# Mappa che associa classi di segmentazione a colori
class_colors = {
    "Branch 1": [1, 0, 0],  # rosso
    "Tree": [0, 1, 0],      # verde
}

# colora i punti in base alla loro classe di segmentazione
new_colors = np.ones_like(points) * 0.5  # default grigio

figures = data.get("figures", [])

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
        new_colors[indices] = color

# Siccome, alcuni oggetti sono spezzati in più figure
# raggruppiamo tutte le figure per objectKey 

# 'defaultdict(list)' è come un dict normale ma se accedi 
# a una chiave che non esiste, viene automaticamente creata 
# con un valore di default (in questo caso lista vuota).
# utilizzo questa mappa, per associare ad un obj_key, gli 
# indici dei suoi punti
object_to_indices = defaultdict(list) 

for fig in data["figures"]:
    obj_key = fig["objectKey"]
    indices = fig["geometry"]["indices"]
    # estende la lista associata all'objectKey corrente
    object_to_indices[obj_key].extend(indices) 

# visualizziamo la pointcloud segmentata
segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(points)
segmented_pcd.colors = o3d.utility.Vector3dVector(new_colors)
print("[INFO] Visualizing segmented point cloud...")
o3d.visualization.draw_geometries([segmented_pcd]) # commenta via se non serve

# approssimo il tronco principale
tree_pc_lineset = []
tree_points = None
tree_dir = None
for branch_obj in data["objects"]:
    # recupero i punti dei rami
    if branch_obj["classTitle"] != "Tree":
        continue

    tree_indices = object_to_indices[branch_obj["key"]]
    tree_points = points[tree_indices]
    tree_colors = colors[tree_indices]

    res = pcpp.approximate_branch(tree_points, tree_colors)
    tree_segments, _, tree_dir, centers, _ = res

    lines = [[i, i+1] for i in range(len(centers)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[0,0,0] for _ in lines]) 
    tree_pc_lineset.append(line_set) 

# approssimo i rami e calcolo le feature
branch_linesets = []
branch_pc_linesets = []
centers = []
for branch_obj in data["objects"]:
    # recupero i punti dei rami
    if branch_obj["classTitle"] != "Branch 1":
        continue
    all_indices = object_to_indices[branch_obj["key"]]
    branch_points = points[all_indices]
    branch_colors = colors[all_indices]

    res = pcpp.approximate_branch(branch_points, branch_colors)
    branch_segments, color_segments, principal_component, centers, pc_line = res

    branch_pc_linesets.append(pc_line)
    # Crea line set per visualizzare i segmenti del ramo
    # Mi basta connettere i centroidi calcolati. Ad es:
    # - centers = [c0, c1, c2, c3, c4]
    # - lines = [[0,1], [1,2], [2,3], [3,4]]
    lines = [[i, i+1] for i in range(len(centers)-1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[1,0,1] for _ in lines]) 
    branch_linesets.append(line_set)

    print(f"=== feature ramo {branch_obj["key"]}===")
    # FIXME: qua sto passando il principal component dell'intero tronco ... questo mi da un vettore che fa schifo.
    # Piuttosto dovrei approssimare anche il tronco con una polilinea, e usare come tree_dir il vettore associato al segmento
    # del tronco il cui il ramo ricade.
    features = pcpp.compute_branch_features(branch_segments, principal_component, tree_points, tree_dir, color_segments)
    pprint(features)

cylinders = []
for ls in branch_linesets:
    cylinders.extend(visualization_stuff.lineset_to_cylinders(ls, radius=0.01, color=[1,1,0]))

o3d.visualization.draw_geometries([segmented_pcd] + cylinders + branch_pc_linesets + tree_pc_lineset)


def create_cut_plane(principal_component, threshold_point, size=0.15):
    """
    Crea un piano ortogonale al principal_component e centrato su threshold_point.
    """
    # Normalizziamo il vettore principale
    n = principal_component / np.linalg.norm(principal_component)

    # Creiamo un piano quadrato molto sottile (0.001 lungo la normale)
    plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.001)

    # Spostiamo il piano in modo che il centro sia in origine
    vertices = np.asarray(plane.vertices)
    vertices -= vertices.mean(axis=0)
    plane.vertices = o3d.utility.Vector3dVector(vertices)

    # Ruotiamo il piano in modo che la normale Z diventi n
    z = np.array([0,0,1])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    if s < 1e-6:
        R = np.eye(3)
    else:
        c = np.dot(z, n)
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c)/(s**2))
    plane.rotate(R, center=(0,0,0))

    # Trasliamo il piano alla soglia
    plane.translate(threshold_point)
    plane.paint_uniform_color([0,0,1])  # blu

    return plane

def color_branch_cut_10_percent(branch_points, branch_colors=None):
    center = branch_points.mean(axis=0)
    points_centered = branch_points - center
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_component = eigvecs[:, np.argmax(eigvals)]
    principal_component /= np.linalg.norm(principal_component)

    proj = points_centered.dot(principal_component)
    min_proj, max_proj = proj.min(), proj.max()
    # taglio ad altezze random
    whatever = 0.2 + random.random() * 0.5
    threshold = min_proj + whatever * (max_proj - min_proj)

    mask = proj > threshold
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(branch_points)

    if branch_colors is None:
        colors = np.ones_like(branch_points) * 0.0
        colors[:,1] = 1.0
    else:
        colors = np.array(branch_colors)

    colors[mask] = np.array([0.5,0.5,0.5])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    threshold_point = center + principal_component * threshold
    cut_plane = create_cut_plane(principal_component, threshold_point)

    return pcd, cut_plane


# ========================================
# Creazione point cloud dei rami tagliati
# ========================================
cut_branch_pcds = []
cut_planes = []

for branch_obj in data["objects"]:
    if branch_obj["classTitle"] != "Branch 1":
        continue

    # poto solo il 40% dei rami
    if random.random() < 0.6:
        continue
    
    all_indices = object_to_indices[branch_obj["key"]]
    branch_points = points[all_indices]
    branch_colors = colors[all_indices]

    pcd_cut, plane = color_branch_cut_10_percent(branch_points, branch_colors)
    cut_branch_pcds.append(pcd_cut)
    cut_planes.append(plane)

o3d.visualization.draw_geometries([segmented_pcd] + cylinders + branch_pc_linesets + tree_pc_lineset + cut_branch_pcds + cut_planes)