"""
Segmentación multi-etapa de una nube de puntos industrial (xyz + rgb + normales).

Etapas
  1. Superficies horizontales (piso, techo, plataformas): histograma de z sobre puntos con
     normal vertical + DBSCAN para separar superficies disjuntas.
  2. Superficies verticales (paredes / paneles): RANSAC iterativo con consistencia de normales.
  3. Descriptores locales a partir de las normales (matriz de segundo momento): distingue
     formas "extruidas" (tuberías, columnas, vigas) y entrega su eje.
  4. Clustering de elementos extruidos: DBSCAN por tramos, agrupados por orientación del eje
     (vertical / X / Y / diagonal). Cada tramo se clasifica con orientación, color y forma de
     la sección (armónicos del ángulo de las normales).
  5. Clustering de los puntos restantes (DBSCAN) y clasificación por dimensiones:
     persona, vehículo, equipo, accesorio.
  6. Suavizado de etiquetas por votación de vecinos y exportación a PLY.
"""

import argparse
import json
import time
import numpy as np
import open3d as o3d
from matplotlib.colors import rgb_to_hsv
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------------------------
# Clases y Colores
# ----------------------------------------------------------------------------------------
UNLABELED, FLOOR, ROOF, PLATFORM, WALL, COLUMN, BEAM, BRACE, PIPE, EQUIPMENT, PERSON, VEHICLE, SMALL = range(13)

CLASSES = {
    UNLABELED: ("sin clasificar", (150, 150, 150)),
    FLOOR: ("piso", (30, 60, 230)),
    ROOF: ("techo", (0, 190, 210)),
    PLATFORM: ("plataforma / entrepiso", (140, 70, 210)),
    WALL: ("pared / panel", (70, 120, 175)),
    COLUMN: ("columna", (0, 170, 0)),
    BEAM: ("viga", (255, 140, 0)),
    BRACE: ("arriostre (diagonal)", (240, 230, 0)),
    PIPE: ("tubería", (225, 0, 0)),
    EQUIPMENT: ("equipo / tanque", (255, 0, 255)),
    PERSON: ("persona", (255, 255, 255)),
    VEHICLE: ("vehículo", (140, 80, 20)),
    SMALL: ("accesorio / equipo menor", (255, 150, 170))}

MAX_PIPE_WIDTH = 0.7  # m: una sección más ancha se considera perfil estructural, no tubería

def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)

# ----------------------------------------------------------------------------------------
# E/S
# ----------------------------------------------------------------------------------------
def load_cloud(path):
    pcd = o3d.io.read_point_cloud(path)
    P = np.asarray(pcd.points).astype(np.float64)
    N = np.asarray(pcd.normals).astype(np.float64)
    C = np.clip(np.asarray(pcd.colors) * 255.0, 0, 255).astype(np.uint8)
    return P, C, N

def write_ply(path, P, rgb, N, labels):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                   ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                   ("label", "u1")])
    arr = np.empty(len(P), dtype=dt)
    arr["x"], arr["y"], arr["z"] = P[:, 0], P[:, 1], P[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr["nx"], arr["ny"], arr["nz"] = N[:, 0], N[:, 1], N[:, 2]
    arr["label"] = labels
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment segmentacion por clases: ver leyenda en el reporte\n"
        f"element vertex {len(P)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar label\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())

# ----------------------------------------------------------------------------------------
# Etapa 1: Superficies Horizontales
# ----------------------------------------------------------------------------------------
def horizontal_layers(P, N, spacing, labels):
    n = len(P)
    cand = np.abs(N[:, 2]) > 0.9
    z = P[cand, 2]
    bw = 0.05
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + bw, bw))
    hs = uniform_filter1d(h.astype(float), 3)
    pk, _ = find_peaks(hs, height=0.003 * n, distance=4)
    peaks = e[pk] + bw / 2
    log(f"capas horizontales (z): {np.round(peaks, 2).tolist()}")
    floor_z, roof_z = peaks.min(), peaks.max()
    for z0 in peaks:
        if z0 <= floor_z + 0.6:
            cls, min_pts = FLOOR, int(0.0015 * n)
        elif z0 >= roof_z - 0.5:
            cls, min_pts = ROOF, int(0.0015 * n)
        else:
            cls, min_pts = PLATFORM, int(0.0025 * n)
        idx = np.where(cand & (np.abs(P[:, 2] - z0) < 0.08) & (labels == UNLABELED))[0]
        if len(idx) < min_pts:
            continue
        cl = DBSCAN(eps=max(0.25, 7 * spacing), min_samples=10, n_jobs=-1).fit_predict(P[idx])
        for c in np.unique(cl[cl >= 0]):
            members = idx[cl == c]
            if len(members) < min_pts:
                continue
            if cls == PLATFORM and np.ptp(P[members, :2], axis=0).min() < 1.0:
                continue  # franja delgada (p. ej. parte superior de una tubería): no es plataforma
            labels[members] = cls
    return floor_z, roof_z

# ----------------------------------------------------------------------------------------
# Etapa 2: Superficies Verticales
# ----------------------------------------------------------------------------------------
def vertical_planes(P, N, spacing, labels, max_planes=10):
    n = len(P)
    idx = np.where((labels == UNLABELED) & (np.abs(N[:, 2]) < 0.2))[0]
    found = 0
    for _ in range(max_planes):
        if len(idx) < 0.002 * n:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx]))
        model, inl = pc.segment_plane(distance_threshold=0.04, ransac_n=3, num_iterations=2000)
        a, b, c, d = model
        inl = np.asarray(inl)
        sel = idx[inl]
        ok = (abs(c) < 0.15) & (np.abs(N[sel] @ np.array([a, b, c])) > 0.95)
        sel_ok = sel[ok]
        idx = np.delete(idx, inl)  # se retiran para avanzar
        if len(sel_ok) < 0.003 * n:
            continue
        cl = DBSCAN(eps=max(0.25, 7 * spacing), min_samples=10, n_jobs=-1).fit_predict(P[sel_ok])
        for k in np.unique(cl[cl >= 0]):
            m = sel_ok[cl == k]
            if len(m) < 0.003 * n:
                continue
            pts = P[m]
            u = np.array([-b, a, 0.0])
            u /= max(np.linalg.norm(u), 1e-9)
            wu = np.ptp(pts @ u)
            hz = np.ptp(pts[:, 2])
            if wu * hz >= 5.0 and min(wu, hz) >= 1.2:
                labels[m] = WALL
                found += 1
    log(f"paredes/paneles verticales encontrados: {found}")

# ----------------------------------------------------------------------------------------
# Etapa 3: Descriptores a Partir de Normales
# ----------------------------------------------------------------------------------------
def normal_descriptors(P, N, k, chunk=150000):
    """Matriz M = media(n n^T) en la vecindad. Autovalores normalizados a<=b<=c y eje = vector
    propio del menor. Plano: (0,0,1). Forma extruida (cilindro, prisma): (0,.5,.5), eje = extrusión."""
    tree = cKDTree(P)
    n = len(P)
    ev = np.empty((n, 3), np.float32)
    axis = np.empty((n, 3), np.float32)
    for s in range(0, n, chunk):
        _, idx = tree.query(P[s:s + chunk], k=k, workers=-1)
        nb = N[idx]
        M = np.einsum("mki,mkj->mij", nb, nb) / k
        w, v = np.linalg.eigh(M)
        w = np.maximum(w, 0)
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        ev[s:s + chunk] = w
        axis[s:s + chunk] = v[:, :, 0]
    return ev, axis, tree

def coarse_descriptors(P, N, spacing, k=60):
    """Mismo descriptor a escala gruesa (vecindad ~25 cm): necesario para perfiles anchos
    (columnas, vigas) cuyas caras parecen planas a escala fina. Se calcula sobre un
    representante por voxel y se propaga al resto por vecino más cercano."""
    vox = 2.0 * spacing
    key = np.floor((P - P.min(0)) / vox).astype(np.int64)
    key = key[:, 0] + key[:, 1] * 100003 + key[:, 2] * 10000019
    _, rep = np.unique(key, return_index=True)
    R, NR = P[rep], N[rep]
    tree = cKDTree(R)
    _, idx = tree.query(R, k=k, workers=-1)
    nb = NR[idx]
    M = np.einsum("mki,mkj->mij", nb, nb) / k
    w, v = np.linalg.eigh(M)
    w = np.maximum(w, 0)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    axis_r = v[:, :, 0]
    _, nn = tree.query(P, k=1, workers=-1)
    return w[nn].astype(np.float32), axis_r[nn].astype(np.float32)

# ----------------------------------------------------------------------------------------
# Etapa 4: Tramos Extruidos (Tuberías, Columnas, Vigas, Arriostres)
# ----------------------------------------------------------------------------------------
def section_harmonics(normals, axis_dir):
    a = axis_dir / np.linalg.norm(axis_dir)
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = np.arctan2(normals @ v, normals @ u)
    h2 = abs(np.mean(np.exp(2j * th)))
    h4 = abs(np.mean(np.exp(4j * th)))
    return h2, h4

def extruded_runs(P, N, C, ev, axis, ev_c, axis_c, spacing, labels):
    n = len(P)
    hsv = rgb_to_hsv(C / 255.0)
    green = (hsv[:, 0] > 0.22) & (hsv[:, 0] < 0.45) & (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.2)
    ext_f = (ev[:, 0] < 0.07) & (ev[:, 1] > 0.15)
    ext_c = (ev_c[:, 0] < 0.08) & (ev_c[:, 1] > 0.20)
    # a escala gruesa gana (perfiles anchos); si no, se usa la escala fina (tubos delgados)
    axis = np.where(ext_c[:, None], axis_c, axis)
    ext = (ext_f | ext_c) & (labels == UNLABELED)
    az, ax, ay = np.abs(axis[:, 2]), np.abs(axis[:, 0]), np.abs(axis[:, 1])
    groups = {
        "Z": ext & (az > 0.92),
        "X": ext & (ax > 0.92),
        "Y": ext & (ay > 0.92),
        "D": ext & (az <= 0.92) & (ax <= 0.92) & (ay <= 0.92)}
    eps = 3.5 * spacing
    ms = max(6, int(8 * n / 1e6))
    min_len = 0.5
    counts = {PIPE: 0, COLUMN: 0, BEAM: 0, BRACE: 0}
    for g, mask in groups.items():
        idx = np.where(mask)[0]
        log(f"  grupo de eje {g}: {len(idx)} puntos extruidos")
        if len(idx) < ms:
            continue
        cl = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1).fit_predict(P[idx])
        for c in np.unique(cl[cl >= 0]):
            m = idx[cl == c]
            if len(m) < 60 * max(1, n / 1e6):
                continue
            pts = P[m]
            a = axis[m].mean(0)
            # el eje es un vector sin signo: se alinea con el primer eje principal del tramo
            a = np.linalg.svd(axis[m].astype(np.float64), full_matrices=False)[2][0]
            length = np.ptp(pts @ a)
            if length < min_len:
                continue
            perp = pts - np.outer(pts @ a, a)
            width = np.sqrt(np.max(np.linalg.eigvalsh(np.cov(perp.T))) * 12)
            h2, h4 = section_harmonics(N[m], a)
            gfrac = green[m].mean()
            cylindrical = (h4 < 0.45) and (h2 < 0.5) and width <= MAX_PIPE_WIDTH
            if gfrac > 0.4 or not cylindrical:
                cls = {"Z": COLUMN, "X": BEAM, "Y": BEAM, "D": BRACE}[g]
            else:
                cls = PIPE
            if width > 1.5:  # demasiado ancho para ser un perfil lineal
                continue
            labels[m] = cls
            counts[cls] += 1
    log(f"tramos: tuberías={counts[PIPE]} columnas={counts[COLUMN]} vigas={counts[BEAM]} arriostres={counts[BRACE]}")

def voxel_linear_members(P, N, C, labels, vs=0.10, min_len=1.5):
    """Etapa 4b: elementos largos de sección ancha (columnas y vigas de perfil grande).
    Se voxeliza lo que queda sin clase y se buscan tramos contiguos de vóxeles ocupados a lo
    largo de cada eje. Un vóxel es 'lineal' si pertenece a un tramo largo en UN solo eje; si lo
    es en dos o más, es parte de una superficie y se descarta."""
    from scipy import ndimage as ndi
    idx = np.where(labels == UNLABELED)[0]
    if len(idx) == 0:
        return
    ijk = np.floor((P[idx] - P.min(0)) / vs).astype(np.int64)
    dims = ijk.max(0) + 3
    ijk += 1
    occ = np.zeros(dims, bool)
    occ[tuple(ijk.T)] = True
    long_ax = []
    for ax in range(3):
        st = np.zeros((3, 3, 3), bool)
        sl = [1, 1, 1]
        sl[ax] = slice(None)
        st[tuple(sl)] = True
        closed = ndi.binary_closing(occ, structure=st) | occ  # puentea huecos de 1 vóxel
        lab, nl = ndi.label(closed, structure=st)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        long_ax.append((sizes[lab] * vs >= min_len) & closed)
    Lx, Ly, Lz = long_ax
    lin = {0: Lx & ~Ly & ~Lz, 1: Ly & ~Lx & ~Lz, 2: Lz & ~Lx & ~Ly}
    hsv = rgb_to_hsv(C / 255.0)
    green = (hsv[:, 0] > 0.22) & (hsv[:, 0] < 0.45) & (hsv[:, 1] > 0.25) & (hsv[:, 2] > 0.2)
    n_new = {COLUMN: 0, BEAM: 0, PIPE: 0}
    for ax, vox in lin.items():
        lab, nl = ndi.label(vox, structure=np.ones((3, 3, 3), bool))
        pl = lab[tuple(ijk.T)]
        axis_dir = np.eye(3)[ax]
        for c in np.unique(pl[pl > 0]):
            m = idx[pl == c]
            if len(m) < 80:
                continue
            ext_ax = np.ptp(P[m, ax])
            if ext_ax < min_len:
                continue
            h2, h4 = section_harmonics(N[m], axis_dir)
            cylindrical = (h4 < 0.45) and (h2 < 0.5)
            perp_axes = [a for a in range(3) if a != ax]
            width = np.ptp(P[m][:, perp_axes], axis=0).max()
            if cylindrical and green[m].mean() <= 0.4 and width <= MAX_PIPE_WIDTH:
                cls = PIPE                      # sección circular, delgada y no es acero verde
            else:
                cls = COLUMN if ax == 2 else BEAM
            labels[m] = cls
            n_new[cls] += 1
    log(f"E4b elementos largos por ocupación de vóxeles: columnas={n_new[COLUMN]} "
        f"vigas={n_new[BEAM]} tuberías={n_new[PIPE]}")

def grow_runs(P, spacing, labels, n_iter=4):
    """Crecimiento de regiones: los puntos sin clase junto a un tramo (p. ej. las caras planas
    de una columna ancha) heredan la clase del tramo."""
    run_classes = [COLUMN, BEAM, BRACE, PIPE]
    for _ in range(n_iter):
        src = np.where(np.isin(labels, run_classes))[0]
        free = np.where(labels == UNLABELED)[0]
        if len(src) == 0 or len(free) == 0:
            break
        d, j = cKDTree(P[src]).query(P[free], k=1, workers=-1)
        take = d < 2.5 * spacing
        if not take.any():
            break
        labels[free[take]] = labels[src[j[take]]]

# ----------------------------------------------------------------------------------------
# Etapa 5: Objetos Compactos Restantes
# ----------------------------------------------------------------------------------------
def isolated(P, m, labels, radius=0.25, max_touch=0.25):
    """Una persona o un vehículo es un objeto aislado: pocos de sus puntos tocan estructura
    (columnas, vigas, tuberías, equipos) fuera del propio objeto."""
    others = np.where(~np.isin(labels, [UNLABELED, FLOOR, ROOF]) & (labels != 0))[0]
    others = others[~np.isin(others, m)]
    if len(others) == 0:
        return True
    d, _ = cKDTree(P[others]).query(P[m], k=1, workers=-1)
    return np.mean(d < radius) < max_touch

def compact_objects(P, C, spacing, labels, floor_z, color_w=0.3, detect_mobile=False):
    n = len(P)
    idx = np.where(labels == UNLABELED)[0]
    if len(idx) == 0:
        return
    # clustering multi-atributo: posición (m) + color (RGB en [0,1] ponderado)
    feat = np.hstack([P[idx], color_w * C[idx] / 255.0])
    cl = DBSCAN(eps=4.5 * spacing, min_samples=max(8, int(10 * n / 1e6)), n_jobs=-1).fit_predict(feat)
    stats = {PERSON: 0, VEHICLE: 0, EQUIPMENT: 0, SMALL: 0}
    for c in np.unique(cl[cl >= 0]):
        m = idx[cl == c]
        if len(m) < 30 * max(1, n / 1e6):
            continue
        pts = P[m]
        dx, dy, dz = np.ptp(pts, axis=0)
        foot = sorted([dx, dy], reverse=True)
        zb = pts[:, 2].min() - floor_z
        # personas y vehículos: solo si se activa --detect-mobile. En esta planta la revisión
        # visual de los candidatos mostró que eran equipos (falsos positivos).
        if detect_mobile and 1.4 <= dz <= 2.1 and foot[0] <= 0.9 and zb < 0.4 \
                and len(m) >= 300 * max(1, n / 1e6) and isolated(P, m, labels):
            cls = PERSON
        elif detect_mobile and 1.2 <= dz <= 3.2 and 2.5 <= foot[0] <= 7.0 and 1.4 <= foot[1] <= 2.8 \
                and zb < 0.5 and isolated(P, m, labels):
            cls = VEHICLE
        elif max(dx, dy, dz) >= 1.0 or len(m) >= 3000 * max(1, n / 1e6):
            cls = EQUIPMENT
        else:
            cls = SMALL
        labels[m] = cls
        stats[cls] += 1
        if cls in (PERSON, VEHICLE):
            log(f"  candidato {CLASSES[cls][0]}: centro={pts.mean(0).round(2).tolist()} "
                f"dims={np.round([dx, dy, dz], 2).tolist()} n={len(m)}")
    log(f"objetos compactos: personas={stats[PERSON]} vehículos={stats[VEHICLE]} "
        f"equipos={stats[EQUIPMENT]} accesorios={stats[SMALL]}")

# ----------------------------------------------------------------------------------------
# Etapa 6: Suavizado por Votación
# ----------------------------------------------------------------------------------------
def smooth_labels(tree, P, labels, k=15, n_iter=2):
    for _ in range(n_iter):
        _, idx = tree.query(P, k=k, workers=-1)
        nb = labels[idx]
        votes = np.zeros((len(P), len(CLASSES)), np.int16)
        for c in range(1, len(CLASSES)):  # las etiquetas 0 no votan
            votes[:, c] = (nb == c).sum(1)
        best = votes.argmax(1)
        bestv = votes.max(1)
        # los sin clasificar adoptan la mayoría; los demás solo cambian con mayoría clara (>=60 %)
        change = ((labels == UNLABELED) & (bestv >= 3)) | ((labels != UNLABELED) & (bestv >= 0.6 * k) & (best != labels))
        labels = np.where(change, best, labels).astype(np.uint8)
    return labels

# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--stats", default=None)
    ap.add_argument("--detect-mobile", action="store_true",
                    help="activa la detección heurística de personas y vehículos")
    args = ap.parse_args()

    P, C, N = load_cloud(args.input)
    n = len(P)
    log(f"nube cargada: {n} puntos")
    tree0 = cKDTree(P)
    spacing = float(np.median(tree0.query(P[:200000], k=2, workers=-1)[0][:, 1]))
    log(f"espaciado mediano: {spacing:.4f} m")

    labels = np.zeros(n, np.uint8)
    floor_z, roof_z = horizontal_layers(P, N, spacing, labels)
    log(f"E1 horizontales: piso={np.sum(labels == FLOOR)} techo={np.sum(labels == ROOF)} plataformas={np.sum(labels == PLATFORM)}")
    vertical_planes(P, N, spacing, labels)
    log(f"E2 verticales: paredes={np.sum(labels == WALL)}")

    k = int(np.clip(50 * n / 1e6, 40, 100))
    ev, axis, tree = normal_descriptors(P, N, k)
    log("E3 descriptores de normales listos")
    ev_c, axis_c = coarse_descriptors(P, N, spacing)
    log("E3b descriptores a escala gruesa listos")
    extruded_runs(P, N, C, ev, axis, ev_c, axis_c, spacing, labels)
    voxel_linear_members(P, N, C, labels)
    grow_runs(P, spacing, labels)
    log("E4 tramos extruidos listos")
    compact_objects(P, C, spacing, labels, floor_z, detect_mobile=args.detect_mobile)
    labels = smooth_labels(tree, P, labels)
    log("E6 suavizado listo")

    rgb = np.zeros((n, 3), np.uint8)
    for c, (_, col) in CLASSES.items():
        rgb[labels == c] = col
    write_ply(args.output, P, rgb, N, labels)
    log(f"exportado: {args.output}")

    stats = {CLASSES[c][0]: int((labels == c).sum()) for c in CLASSES}
    for name, v in stats.items():
        print(f"  {name:28s} {v:9d}  {100 * v / n:5.1f} %")
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump({"n_points": n, "spacing_m": spacing, "classes": stats}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()