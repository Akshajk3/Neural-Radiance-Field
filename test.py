import torch
import numpy as np
import matplotlib.pyplot as plt
from fast_nerf_test import render_rays, FastNerf

# Load the model
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(device)

model = torch.load('nerf_model.pth', map_location=device)
model.to(device)
model.eval()

# Load test data
testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))

# Set parameters
H, W = 400, 400  # Image height and width
img_index = 0  # Index of the image to generate

# Prepare ray origins and directions
ray_origins = testing_dataset[img_index * H * W: (img_index + 1) * H * W, :3]
ray_directions = testing_dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

# Reduce memory usage by processing rays in smaller batches
batch_size = 2048  # Adjust based on your GPU memory, smaller if still encountering OOM

regenerated_px_values = []
with torch.no_grad():
    for i in range(0, ray_origins.shape[0], batch_size):
        batch_origins = ray_origins[i:i + batch_size].to(device)
        batch_directions = ray_directions[i:i + batch_size].to(device)

        batch_output = render_rays(model, batch_origins, batch_directions, hn=2, hf=6, nb_bins=192)

        regenerated_px_values.append(batch_output.cpu())

# Concatenate the results
regenerated_px_values = torch.cat(regenerated_px_values, dim=0)

# Visualize or save the result
plt.figure()
plt.imshow(regenerated_px_values.numpy().reshape(H, W, 3).clip(0, 1))
plt.axis('off')
plt.savefig('generated_images/generated_image.png', bbox_inches='tight')
plt.show()