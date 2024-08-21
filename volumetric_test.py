import torch
import numpy as np
import matplotlib.pyplot as plt
import mcubes  # Import the marching cubes library
from fast_nerf_test import render_rays, FastNerf

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('nerf_model.pth', map_location=device)
model.to(device)
model.eval()

# Define the 3D grid (you can adjust the resolution)
grid_resolution = 128
x = torch.linspace(-1, 1, grid_resolution)
y = torch.linspace(-1, 1, grid_resolution)
z = torch.linspace(-1, 1, grid_resolution)
xyz = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3).to(device)

# Batch size for processing the grid points
batch_size = 2048

# Initialize volume grid for storing density
density_grid = np.zeros((grid_resolution, grid_resolution, grid_resolution), dtype=np.float32)

# Placeholder direction vector (e.g., pointing in the z-direction)
direction = torch.tensor([0, 0, 1], device=device).repeat(batch_size, 1)

# Populate the density grid by querying the model
with torch.no_grad():
    for i in range(0, xyz.shape[0], batch_size):
        batch_xyz = xyz[i:i + batch_size]

        # Adjust direction batch size if last batch is smaller
        if batch_xyz.shape[0] < batch_size:
            direction = torch.tensor([0, 0, 1], device=device).repeat(batch_xyz.shape[0], 1)

        sigma, _ = model(batch_xyz, direction)

        # Convert sigma to numpy and update density grid
        sigma = sigma.cpu().numpy()

        # Calculate the indices in the 3D grid
        idx_x = (i // (grid_resolution ** 2)) % grid_resolution
        idx_y = (i // grid_resolution) % grid_resolution
        idx_z = i % grid_resolution

        # Reshape sigma and assign to the appropriate slice in the density grid
        density_grid[idx_x:idx_x + sigma.shape[0] // (grid_resolution ** 2),
        idx_y:idx_y + sigma.shape[0] // grid_resolution,
        idx_z:idx_z + sigma.shape[0]] = sigma.reshape(
            density_grid[idx_x:idx_x + sigma.shape[0] // (grid_resolution ** 2),
            idx_y:idx_y + sigma.shape[0] // grid_resolution,
            idx_z:idx_z + sigma.shape[0]].shape
        )

# Extract the mesh using marching cubes
vertices, faces = mcubes.marching_cubes(density_grid, threshold=0.5)

# Save the mesh to an .obj file
mcubes.export_obj(vertices, faces, "generated_model.obj")

# Optional: Visualize the result
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Poly3DCollection(vertices[faces], facecolors='w', edgecolors='k'))
ax.set_xlim([0, grid_resolution])
ax.set_ylim([0, grid_resolution])
ax.set_zlim([0, grid_resolution])
plt.show()