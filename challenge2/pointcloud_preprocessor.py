import numpy as np
import open3d as o3d 

seg_polilinea = 5     # numero di segmenti della polilinea con cui si approssima ogni ramo
k = seg_polilinea + 1 # questo è il numero di intervalli contenenti punti del ramo che utilizziamo per calcolare la polilinea

# =========================================================
# PCA per ogni ramo 
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
#   - La covarianza misura come due variabili (ad esempio X e Y) cambiano insieme.
#       - se cov(x, y) > 0 -> se x cresce, y cresce; se cov(x,y) < 0 -> se x cresce y, y decresce
#       - la correlazione è una versione “normalizzata” della covarianza.
# - Trova gli autovettori della matrice di covarianza:
#   - A quanto pare si può dimostrare che l'autovalore più grande definisce un autovettore
#     che rappresenta proprio è la direzione lungo cui i punti variano di più → cioè l'asse principale del ramo;
#   - gli altri due sono direzioni ortogonali minori (spessore e profondità).
# 
# Una volta trovata l'asse principale di ogni ramo, per approssimare quest'ultimi possiamo
# - dividere l'asse in parti uguali
# - proiettare ogni punto del ramo sull'asse e vedere in che sezione finisce
# - ... TODO: finisci

def PCA(points, center):
    points_centered = points - center
    # np.cov() si aspetta una matrice dove ogni riga è una variabile e ogni colonna è un'osservazione.
    # Ovvero una shape: (3, num_punti)
    # ma noi abbiamo un array di punti con shape: (num_punti, 3), e quindi facciamo una trasposta
    cov = np.cov(points_centered.T) # .T == .transpose()
    eigvals, eigvecs = np.linalg.eig(cov)
    # eigvecs ha shape (3, num_eigenvecs), è quindi una matrice di vettori colonna
    # prendiamo il vettore colonna associato all'autovalore più grande
    principal_component = eigvecs[:, np.argmax(eigvals)]  # direzione principale

    return points_centered, principal_component

def approximate_branch(branch_points, branch_colors):
    """
    questa funzione, dati gli oggetti che descrivono i rami nella pointcloud segmentata,
    approssima i rami con una polilinea

    Parametri:
        - branch_points: i punti che rappresentano un ramo
        - branch_colors: colori dei punti del ramo
        
    Ritorna:
        - branch_segments: lista dei punti del ramo suddivisi per segmenti
        - color_segments: lista dei colori del ramo suddivisi per segmenti 
        - principal_component: asse che approssima il meglio possibile il ramo
        - centers: la lista di centroidi che rappresentano gli estremi dei segmenti della polilinea
        - pc_line: la linea del principal_component del ramo (per visualizzazione)
    """
    
    center = branch_points.mean(axis=0)
    points_centered, principal_component = PCA(branch_points, center)
    # ottengo le proiezioni di tutti i punti del ramo sul principal component
    proj = points_centered.dot(principal_component)

    # Calcolo degli estremi della retta del principal component
    pc_start = center + principal_component * proj.min()
    pc_end   = center + principal_component * proj.max()
    pc_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([pc_start, pc_end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    pc_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # blu

    # Suddivisione in k segmenti:
    # - proj.min() → valore minimo lungo l’asse
    # - proj.max() → valore massimo lungo l’asse
    # - np.linspace(start, stop, num) Genera num valori equidistanti tra start e stop.
    edges = np.linspace(proj.min(), proj.max(), k+1)

    centers = []
    branch_segments = []
    color_segments = []
    for i in range(k):
        # Controlliamo quali proiezioni finiscono in quale segmento
        # - mask è un array booleano della stessa lunghezza di proj, dove:
        #   - True = il punto cade nel segmento i-esimo
        #   - False = il punto è fuori dal segmento
        mask = (proj >= edges[i]) & (proj < edges[i+1])
        if np.any(mask):
            # utilizzo la maschera per ottenere i punti del ramo corrispondenti
            # al segmento corrente. Di questi ne calcolo il centroide
            branch_segment_points = branch_points[mask]
            branch_segments.append(branch_segment_points)
            centers.append(branch_segment_points.mean(axis=0))
            color_segment = branch_colors[mask]
            color_segments.append(color_segment)

    centers = np.array(centers)
    if len(centers) < 2:
        return None, None, None  # non posso creare una polilinea con <2 punti

    return branch_segments, color_segments, principal_component, centers, pc_line


def compute_branch_features(branch_segments, branch_dir, tree_points, tree_dir, color_segments):
    """
    Calcola feature utili per un ramo segmentato in sotto-segmenti.

    Parametri:
        - branch_segments: lista di array Nx3, ogni array rappresenta un segmento del ramo
        - branch_dir: principal component del ramo
        - tree_points: point cloud (Nx3) del tronco/capofila principale
        - tree_dir: principal component del tronco
        - branch_colors: array_contenente i colori dei punti del ramo

    Ritorna:
        Un dizionario con:
            - diameters: diametro stimato per ogni segmento
            - colors: colore medio per segmento (se disponibile)
            - branch_length: lunghezza totale del ramo
            - inclination_angle: angolo in gradi tra ramo e tronco
    """

    # ------------------------------
    # 1. DIAMETRO LOCALE PER SEGMENTO
    # ------------------------------
    diameters = []
    seg_centers = []

    for seg in branch_segments:
        if len(seg) < 3:
            diameters.append(0)
            seg_centers.append(seg.mean(axis=0))
            continue

        center = seg.mean(axis=0)
        seg_centers.append(center)
        # PCA locale per stimare diametro
        pts_centered = seg - center
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        # diametro = 2*std_dev (autovalore minore)
        raggio = np.sqrt(np.min(eigvals))
        diameter = 2*raggio
        diameters.append(float(diameter))

    # ------------------------------
    # 2. LUNGHEZZA TOTALE DEL RAMO
    # ------------------------------
    branch_length = 0.0
    seg_centers = np.array(seg_centers)
    for i in range(len(seg_centers) - 1):
        branch_length += np.linalg.norm(seg_centers[i+1] - seg_centers[i])

    # ------------------------------
    # 3. INCLINAZIONE DEL RAMO RISPETTO AL TRONCO
    # ------------------------------

    branch_dir = branch_dir / np.linalg.norm(branch_dir)
    tree_dir = tree_dir / np.linalg.norm(tree_dir)
    dot = branch_dir.dot(tree_dir)
    angle_rad = np.arccos(abs(dot))
    inclination_angle = np.degrees(angle_rad)

    # ------------------------------
    # 4. COLORE MEDIO per segmento (se esiste)
    # ------------------------------

    mean_colors = []
    for seg_colors in color_segments:
        if len(seg_colors) > 0:
            mean_colors.append(seg_colors.mean(axis=0))
        else:
            mean_colors.append(None)

    # ------------------------------
    # OUTPUT
    # ------------------------------
    return {
        "diameters": diameters,
        "branch_length": float(branch_length),
        "inclination_angle": float(inclination_angle),
        "mean_colors": mean_colors
    }

    # df = pd.DataFrame({
    #     "segment_index": list(range(len(branch_segments))),
    #     "diameter": diameters,
    #     "center_x": seg_centers[:, 0],
    #     "center_y": seg_centers[:, 1],
    #     "center_z": seg_centers[:, 2],
    #     "mean_color": colors
    # })

