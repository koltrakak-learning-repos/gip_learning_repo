import open3d as o3d
import numpy as np

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