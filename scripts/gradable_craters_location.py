import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# Load the PGM image in grayscale
image_path = "/home/simson/simson_ws/CMU_Capstone_Project/Lunar_ROADSTER_ws/src/mapping/costmap/V2/gradable_craters.pgm"
costmap = plt.imread(image_path)

# Ensure grayscale normalization (some PGM files are in [0, 255] and others in [0, 1])
if costmap.max() > 1:
    costmap = costmap / 255.0  # Normalize to range [0,1]

# Flip the image vertically so the y-axis starts from the bottom-left
costmap = np.flipud(costmap)

# Threshold to identify black regions (craters)
binary_map = costmap < 0.5  # Consider pixels below 0.5 intensity as "black"

# Label connected components (clusters of black dots)
labeled_map, num_features = ndi.label(binary_map)

# Get the sizes (number of points) of each cluster
cluster_sizes = np.bincount(labeled_map.ravel())[1:]  # Ignore the background (label 0)

# Define a threshold for the minimum cluster size to keep
min_cluster_size = 50  # Adjust this value to suit your needs

# Filter clusters based on the size
large_clusters = cluster_sizes >= min_cluster_size

# Compute centroids of each cluster (only for large clusters)
crater_centroids = []
for i in range(1, num_features + 1):
    if i <= len(large_clusters) and large_clusters[i - 1]:  # Only include large clusters
        centroid = ndi.center_of_mass(binary_map, labeled_map, [i])
        crater_centroids.append(centroid)

# Convert centroids to a numpy array
crater_centroids = np.array(crater_centroids)

# Ensure centroids is not empty and has at least two columns
if crater_centroids.size > 0 and crater_centroids.shape[1] == 1:
    crater_centroids = crater_centroids.reshape(-1, 2)  # Reshape to ensure (x, y) format

# Convert to (x, y) coordinates in the costmap
if crater_centroids.size > 0:
    crater_centroids[:, [0, 1]] = crater_centroids[:, [1, 0]]  # Swap (row, col) -> (x, y) for visualization

    # Convert to world coordinates with origin at bottom-left (0, 0)
    map_height, map_width = costmap.shape
    resolution = 0.01  # Set resolution (meters per pixel)
    origin_x, origin_y = 0, 0  # Start coordinates from (0, 0)

    # Adjust y-coordinate to start from the bottom-left
    # Since the image is flipped, no need to invert y-axis again
    world_crater_centroids = (crater_centroids) * resolution  # No need for origin correction as (0, 0) is the origin now

    # Print centroid coordinates for large craters
    print("Centroids of Gradable Craters (World Coordinates):")
    for i, (wx, wy) in enumerate(world_crater_centroids):
        print(f"Crater {i+1}: X = {wx:.3f} m, Y = {wy:.3f} m")
    
    # Visualization with origin at (0, 0) on both axes (bottom-left as origin for image)
    plt.figure(figsize=(8, 8))
    plt.imshow(costmap, cmap="gray", origin="lower")  # Use 'lower' to start y-axis from bottom
    plt.scatter(crater_centroids[:, 0], crater_centroids[:, 1], c="red", marker="x", label="Gradable Crater Centroids")
    
    # Add labels (C1, C2, ...) to the centroids
    for i, (x, y) in enumerate(crater_centroids):
        plt.text(x, y, f"C{i+1}", color="blue", fontsize=12, ha="center", va="bottom")

    plt.legend()
    plt.title("Gradable Crater Detection and Centroids")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
else:
    print("No Gradable craters found.")
