def cut_branch_at_10_percent(branch_points):
    """
    Taglia un ramo classificato come da tagliare al 10% della lunghezza lungo il principal component.

    Parametri:
        branch_points: np.array Nx3 dei punti del ramo

    Ritorna:
        cut_points: punti rimasti dopo il taglio
        cut_plane: o3d.geometry.TriangleMesh che rappresenta il piano di taglio
    """

    # 1. PCA
    center = branch_points.mean(axis=0)
    points_centered = branch_points - center
    cov = np.cov(points_centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_component = eigvecs[:, np.argmax(eigvals)]
    principal_component /= np.linalg.norm(principal_component)

    # 2. Proiezione dei punti sull'asse principale
    proj = points_centered.dot(principal_component)

    # 3. Soglia per il 10%
    min_proj, max_proj = proj.min(), proj.max()
    threshold = min_proj + 0.1 * (max_proj - min_proj)

    # 4. Maschera dei punti sotto la soglia
    mask = proj <= threshold
    cut_points = branch_points[mask]

    # 5. Creazione del piano di taglio
    # piano passa per la soglia e normale è il principal component
    plane_center = center + principal_component * threshold
    plane = o3d.geometry.TriangleMesh.create_box(width=0.01, height=1.0, depth=1.0)
    # Ruotiamo e trasliamo il piano per farlo coincidere con la soglia
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))  # placeholder, possiamo orientare meglio
    plane.rotate(R, center=(0,0,0))
    plane.translate(plane_center)

    return cut_points, plane