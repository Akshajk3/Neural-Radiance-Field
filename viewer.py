import open3d

import open3d as o3d

pcd = o3d.io.read_point_cloud('lego.ply')

o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)