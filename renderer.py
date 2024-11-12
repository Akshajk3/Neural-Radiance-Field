import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_point_cloud(filename):
    point_cloud = o3d.io.read_point_cloud(filename)
    points = np.asarray(point_cloud.points)

    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
    else:
        colors = np.ones((points.shape[0], 3))

    return points, colors

def guassian_kernal(x, y, sigma=1.0):
    return np.exp(-0.5 * (x**2 + y**2) / sigma**2)

def render_splats(points, colors, image_size=(512, 512), sigma=1.0):
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.float32)
    image_width, image_height = image_size

    for i, point in enumerate(points):
        x, y, z = point

        px = int((x + 1) / 2 * image_width)
        py = int((y + 1) / 2 * image_height)

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                gx, gy = px + dx, py + dy
                if 0 <= gx < image_width and 0 <= gy < image_height:
                    weight = guassian_kernal(dx, dy, sigma)
                    image[gy, gx] += weight * colors[i]
        
    image = np.clip(image, 0, 1)
    return image

def main():
    ply_path = "luigi.ply"
    points, colors = load_point_cloud(ply_path)
    
    rendered_image = render_splats(points, colors, image_size=(512, 512), sigma=1.0)

    plt.imshow(rendered_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()