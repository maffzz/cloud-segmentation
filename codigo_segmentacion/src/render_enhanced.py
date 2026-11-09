#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_INFO = [
    ("Piso", (30/255, 60/255, 230/255)),
    ("Viga", (255/255, 140/255, 0/255)),
    ("Columna", (0/255, 170/255, 0/255)),
    ("Tubería", (225/255, 0/255, 0/255)),
    ("Techo", (0/255, 190/255, 210/255)),
    ("Plataforma / entrepiso", (140/255, 70/255, 210/255)),
    ("Equipo / tanque", (255/255, 0/255, 255/255)),
    ("Accesorio", (255/255, 150/255, 170/255)),
    ("Arriostre", (240/255, 230/255, 0/255)),
    ("Sin clasificar", (150/255, 150/255, 150/255)),
]

def load_ply_np(path, has_label=False):
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if b"end_header" in line:
                break
        dtype = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")
        ]
        if has_label:
            dtype.append(("label", "u1"))
        arr = np.frombuffer(f.read(), dtype=dtype)
    P = np.column_stack([arr["x"], arr["y"], arr["z"]])
    C = np.column_stack([arr["r"], arr["g"], arr["b"]]) / 255.0
    lbl = arr["label"] if has_label else None
    return P, C, lbl

def project_view(P, C, az, el, ax, title=None, s=0.5):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(d, right)
    x, y, depth = P @ right, P @ up, P @ d
    o = np.argsort(depth)
    ax.scatter(x[o], y[o], c=C[o], s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=12, pad=6, fontweight="bold")
    ax.axis("off")

def main():
    orig_path = "FARO_Scan_Full_Sub1_down3_with_normals.ply"
    seg_path = "nube_segmentada_completa.ply"
    out_dir = "codigo_segmentacion/out"
    os.makedirs(out_dir, exist_ok=True)

    print("Cargando nubes...")
    P_orig, C_orig, _ = load_ply_np(orig_path)
    P_seg, C_seg, labels = load_ply_np(seg_path, has_label=True)

    # 1. Comparativa lado a lado
    print("Generando comparativa lado a lado...")
    n_sample = 500000
    idx = np.random.default_rng(42).choice(len(P_orig), min(n_sample, len(P_orig)), replace=False)
    Po_s, Co_s = P_orig[idx], C_orig[idx]
    Ps_s, Cs_s = P_seg[idx], C_seg[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=130)
    project_view(Po_s, Co_s, -60, 25, ax1, title="Nube Original (RGB Real)", s=0.45)
    project_view(Ps_s, Cs_s, -60, 25, ax2, title="Nube Segmentada por Clases Semánticas", s=0.45)

    # Add legend at bottom
    patches = [mpatches.Patch(color=c, label=l) for l, c in CLASS_INFO]
    fig.legend(handles=patches, loc="lower center", ncol=5, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    cmp_path = os.path.join(out_dir, "comparativa_original_vs_segmentado.png")
    plt.savefig(cmp_path, dpi=130)
    plt.close()
    print(f"Guardado: {cmp_path}")

    # 2. Acercamiento: Tuberías y Vigas
    print("Generando acercamiento tuberías y vigas...")
    # Región densa en tuberías: x in [4, 16], y in [-4, 3], z in [0, 4.5]
    mask_tub = (P_seg[:, 0] >= 4.0) & (P_seg[:, 0] <= 16.0) & \
               (P_seg[:, 1] >= -4.0) & (P_seg[:, 1] <= 3.0) & \
               (P_seg[:, 2] >= 0.0) & (P_seg[:, 2] <= 4.5)
    Pt = P_seg[mask_tub]
    Ct = C_seg[mask_tub]
    if len(Pt) > 300000:
        it = np.random.default_rng(42).choice(len(Pt), 300000, replace=False)
        Pt, Ct = Pt[it], Ct[it]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=130)
    project_view(Pt, Ct, -45, 20, ax, title="Detalle A: Trazado de Tuberías (Rojo) y Enrejado de Vigas (Naranja)", s=1.2)
    plt.tight_layout()
    det1_path = os.path.join(out_dir, "detalle_tuberias_vigas.png")
    plt.savefig(det1_path, dpi=130)
    plt.close()
    print(f"Guardado: {det1_path}")

    # 3. Acercamiento: Columnas, Arriostres y Piso
    print("Generando acercamiento columnas y arriostres...")
    mask_col = (P_seg[:, 0] >= 0.0) & (P_seg[:, 0] <= 14.0) & \
               (P_seg[:, 1] >= 2.0) & (P_seg[:, 1] <= 12.0) & \
               (P_seg[:, 2] >= -1.8) & (P_seg[:, 2] <= 5.0)
    Pc = P_seg[mask_col]
    Cc = C_seg[mask_col]
    if len(Pc) > 300000:
        ic = np.random.default_rng(42).choice(len(Pc), 300000, replace=False)
        Pc, Cc = Pc[ic], Cc[ic]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=130)
    project_view(Pc, Cc, -70, 15, ax, title="Detalle B: Columnas (Verde), Arriostres Diagonales (Amarillo) y Piso (Azul)", s=1.2)
    plt.tight_layout()
    det2_path = os.path.join(out_dir, "detalle_columnas_arriostres.png")
    plt.savefig(det2_path, dpi=130)
    plt.close()
    print(f"Guardado: {det2_path}")

    # 4. Acercamiento: Plataforma elevada y Equipos
    print("Generando acercamiento plataformas y equipos...")
    mask_plat = (P_seg[:, 0] >= 5.0) & (P_seg[:, 0] <= 18.0) & \
                (P_seg[:, 1] >= -6.0) & (P_seg[:, 1] <= 2.0) & \
                (P_seg[:, 2] >= 1.0) & (P_seg[:, 2] <= 4.0)
    Pp = P_seg[mask_plat]
    Cp = C_seg[mask_plat]
    if len(Pp) > 300000:
        ip = np.random.default_rng(42).choice(len(Pp), 300000, replace=False)
        Pp, Cp = Pp[ip], Cp[ip]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=130)
    project_view(Pp, Cp, -30, 25, ax, title="Detalle C: Plataforma de Mantenimiento (Morado) y Maquinaria/Equipos (Magenta)", s=1.2)
    plt.tight_layout()
    det3_path = os.path.join(out_dir, "detalle_plataforma_equipos.png")
    plt.savefig(det3_path, dpi=130)
    plt.close()
    print(f"Guardado: {det3_path}")

    print("¡Todas las figuras mejoradas fueron generadas con éxito!")

if __name__ == "__main__":
    main()

