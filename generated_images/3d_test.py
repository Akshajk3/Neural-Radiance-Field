import torch
import numpy as np
from fast_nerf_test import render_rays, FastNerf
import open3d as o3d

import os
os.environ["OMP_NUM_THREADS"] = "1"

# Load the model
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

model = torch.load('nerf_model.pth', map_location=device)
model.to(device)
model.eval()

# Define the 3D grid
x = np.linspace(-2, 2, 128)
y = np.linspace(-2, 2, 128)
z = np.linspace(-2, 2, 128)
xx, yy, zz = np.meshgrid(x, y, z)
grid_points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
grid_points = torch.tensor(grid_points, dtype=torch.float32).to(device)

# Query the NeRF model for densities and colors
densities = []
colors = []
batch_size = 2048
with torch.no_grad():
    for i in range(0, grid_points.shape[0], batch_size):
        batch_points = grid_points[i:i + batch_size]
        batch_densities, batch_colors = model(batch_points)
        densities.append(batch_densities.cpu())
        colors.append(batch_colors.cpu())

# Concatenate the results
densities = torch.cat(densities, dim=0).numpy()
colors = torch.cat(colors, dim=0).numpy()

# Filter points based on density
density_threshold = 0.1  # Adjust this threshold
valid_points = grid_points.cpu().numpy()[densities > density_threshold]
valid_colors = colors[densities > density_threshold]

# Create a point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(valid_points)
point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)

# Save the point cloud to a PLY file
o3d.io.write_point_cloud("generated_model.ply", point_cloud)