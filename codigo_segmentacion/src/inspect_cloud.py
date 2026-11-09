import sys, numpy as np, open3d as o3d

pcd = o3d.io.read_point_cloud(sys.argv[1])
P = np.asarray(pcd.points)
C = np.asarray(pcd.colors)
N = np.asarray(pcd.normals)

print("n =", len(P), "| has colors:", pcd.has_colors(), "| has normals:", pcd.has_normals())
print("min", P.min(0).round(2), "max", P.max(0).round(2), "extent", (P.max(0)-P.min(0)).round(2))
print("centroid", P.mean(0).round(2))
print("colors range", C.min(), C.max(), "mean", C.mean(0).round(3))
print("normal norm mean", np.linalg.norm(N,axis=1).mean().round(3))

for i,a in enumerate("xyz"):
    print(f"|n{a}|>0.9 :", (np.abs(N[:,i])>0.9).mean().round(3))

for i,a in enumerate("xyz"):
    h,e = np.histogram(P[:,i], bins=20)
    print(a, "hist:", h.tolist())

# nearest neighbor spacing
d = np.asarray(pcd.compute_nearest_neighbor_distance())
print("NN dist mean/median/p95", d.mean().round(4), np.median(d).round(4), np.percentile(d,95).round(4))