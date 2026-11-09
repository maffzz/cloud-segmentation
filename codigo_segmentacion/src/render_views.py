#!/usr/bin/env python3
import sys, numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def view(P, C, az, el, ax, title, s=0.4):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])      # hacia la cámara
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(d, right)
    x, y, depth = P @ right, P @ up, P @ d
    o = np.argsort(depth)                                                    # lo más cercano al final
    ax.scatter(x[o], y[o], c=C[o], s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_facecolor("white"); ax.set_title(title); ax.axis("off")

def main(path, out, n_sub=500000):
    pcd = o3d.io.read_point_cloud(path)
    P = np.asarray(pcd.points); C = np.asarray(pcd.colors)
    if len(P) > n_sub:
        i = np.random.default_rng(0).choice(len(P), n_sub, replace=False); P, C = P[i], C[i]
    views = [(-60, 25, "Vista oblicua 1"), (30, 25, "Vista oblicua 2"), (-90, 0, "Alzado"), (0, 0, "Perfil"), (-90, 89, "Planta")]
    for k, (az, el, t) in enumerate(views):
        fig, ax = plt.subplots(figsize=(10, 7), dpi=110)
        view(P, C, az, el, ax, t); plt.tight_layout(); plt.savefig(f"{out}_{k}.png"); plt.close()
    print("ok")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])