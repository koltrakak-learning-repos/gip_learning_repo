import open3d as o3d
import json
import numpy as np
import os
from collections import defaultdict

PCD_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/pointcloud/pc_color_filtered.pcd"
ANN_PATH = "./Vineyard Pointcloud/dataset 2025-10-03 09-46-48/ann/pc_color_filtered.pcd.json"

pcd = o3d.io.read_point_cloud(PCD_PATH)
points = np.asarray(pcd.points)
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
colors = np.ones_like(points) * 0.5  # default grigio

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
        colors[indices] = color

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
    branch_points = points[all_indices]
    branch_counter += 1
    print(f"\n--- Branch {branch_counter} (objectKey={obj["key"]}) ---")
    print("Primi 10 punti:")
    print(branch_points[:10])

print(f"\nTotale rami trovati: {branch_counter}")


# =========================================================
# CREA POINTCLOUD OPEN3D E VISUALIZZA
# =========================================================

# funzioni di visualizzazione che trasformano linee (spesse 1px)
# in cilindri di spessore più grande
def line_to_cylinder(start, end, radius=0.01, color=[1,0,1], resolution=20):
    """
    Crea un cilindro che collega i punti `start` e `end`.
    - start, end: array-like (3,)
    - radius: raggio del cilindro
    - color: lista RGB nel range [0,1]
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

def lineset_to_cylinders(line_set, radius=0.01, color=[1,0,0]):
    cylinders = []
    pts = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    for (i, j) in lines:
        start = pts[i]
        end   = pts[j]
        cyl = line_to_cylinder(start, end, radius=radius, color=color)
        if cyl is not None:
            cylinders.append(cyl)

    return cylinders

segmented_pcd = o3d.geometry.PointCloud()
segmented_pcd.points = o3d.utility.Vector3dVector(points)
segmented_pcd.colors = o3d.utility.Vector3dVector(colors) # commenta via se vuoi i colori originali della pointcloud
print("[INFO] Visualizing segmented point cloud...")
o3d.visualization.draw_geometries([segmented_pcd])

# =========================================================
# PCA per ogni ramo (TODO: aggiungere emoji fuoco)
# =========================================================

# PCA (Principal Component Analisys) è una tecnica che serve a trovare 
# le direzioni principali di variazione dei dati.
# 
# Nel nostro caso abbiamo nuvole di punti 3D che rappresentano un ramo.
#
# In ogni ramo la maggior parte dei punti si distribuisce lungo una 
# direzione principale (l'asse del ramo), e si ha un po' di dispersione
# (spessore del ramo, irregolarità)
#
# La PCA serve a trovare quell’asse principale in modo automatico.
#
# Come funziona, in breve:
# - Prende tutti i punti della tua nuvola.
# - Calcola la media (centro del ramo) e usa il centro come origine del sistema di riferimento
# - Mediante la matrice di covarianza, Calcola la covarianza tra le coordinate X, Y, Z 
#   - cioè quanto le variazioni su un asse sono correlate con le altre.
#                       [var(x), cov(x,y), cov(x,z)]
#   - cov(punti_ramo) = [cov(y,x), var(y), cov(y,z)] 
#                       [cov(z,x), cov(z,y), var(z)]
#   - La covarianza come due variabili (ad esempio X e Y) cambiano insieme.
#       - se cov(x, y) > 0 -> se x cresce, y cresce; se cov(x,y) < 0 -> se x cresce y, y decresce
#       - la correlazione è una versione “normalizzata” della covarianza.
# - Trova gli autovettori della matrice di covarianza:
#   - A quanto pare si può dimostrare che l'autovalore più grande definisce un autovettore
#     che rappresenta proprio è la direzione lungo cui i punti variano di più → cioè l'asse principale del ramo;
#   - gli altri due sono direzioni ortogonali minori (spessore e profondità).


seg_polilinea = 5     # numero di segmenti della polilinea con cui si approssima ogni ramo
k = seg_polilinea + 1 # questo è il numero di intervalli contenenti punti del ramo che utilizziamo per calcolare la polilinea

branch_linesets = []
for obj in data["objects"]:
    # recupero i punti dei rami
    if obj["classTitle"] != "Branch 1":
        continue
    all_indices = object_to_indices[obj["key"]]
    branch_points = points[all_indices]

    if len(branch_points) < 5:
        continue  # troppo pochi punti per PCA

    # PCA
    center = branch_points.mean(axis=0)
    points_centered = branch_points - center
    # np.cov() si aspetta una matrice dove ogni riga è una variabile e ogni colonna è un'osservazione.
    # Ovvero una shape: (3, num_punti)
    # ma noi abbiamo un array di punti con shape: (num_punti, 3), e quindi facciamo una trasposta
    cov = np.cov(points_centered.T) # .T == .transpose()
    eigvals, eigvecs = np.linalg.eig(cov)
    # eigvecs ha shape (3, num_eigenvecs), è quindi una matrice di vettori colonna
    # prendiamo il vettore colonna associato all'autovalore più grande
    principal_component = eigvecs[:, np.argmax(eigvals)]  # direzione principale

    # ottengo le proiezioni di tutti i punti del ramo sul principal component
    proj = points_centered.dot(principal_component)
    # Suddivisione in k segmenti:
    # - proj.min() → valore minimo lungo l’asse
    # - proj.max() → valore massimo lungo l’asse
    # - np.linspace(start, stop, num) Genera num valori equidistanti tra start e stop.
    edges = np.linspace(proj.min(), proj.max(), k+1)

    centers = []
    for i in range(k):
        # Controlliamo quali proiezioni finiscono in quale segmento
        # - mask è un array booleano della stessa lunghezza di proj, dove:
        #   - True = il punto cade nel segmento i-esimo
        #   - False = il punto è fuori dal segmento
        mask = (proj >= edges[i]) & (proj < edges[i+1])
        if np.any(mask):
            # utilizzo la maschera per ottenere i punti del ramo corrispondenti
            # al segmento corrente. Di questi ne calcolo il centroide

            # TODO: su questo branch segment posso calcolare le feature interessanti
            # - Diametro
            # - colore
            # - lunghezza ramo (ci dice numero di gemme)
            # - inclinazione del ramo rispetto al cordone principale (angolo)
            branch_segment = branch_points[mask]
            centers.append(branch_segment.mean(axis=0))

    centers = np.array(centers)
    if len(centers) < 2:
        continue  # non posso creare una polilinea con <2 punti

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

cylinders = []
for ls in branch_linesets:
    cylinders.extend(lineset_to_cylinders(ls, radius=0.01, color=[1,0,1]))

# Visualizzazione

o3d.visualization.draw_geometries([segmented_pcd] + cylinders)
o3d.visualization.draw_geometries(cylinders)