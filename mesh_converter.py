import open3d as o3d
import numpy as np

pcd_raw = o3d.io.read_point_cloud("dense_cleaned.ply")

cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
pcd = pcd_raw.select_by_index(ind)

pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()

o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0]

rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
mesh.rotate(rotation, center = (0, 0, 0))

o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

o3d.io.write_triangle_mesh("dense_cleaned.obj", mesh)