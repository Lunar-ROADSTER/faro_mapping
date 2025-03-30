import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2

# Function to calculate centroids of gradable craters
def calculate_centroids(image_path):
    # Load the PGM image in grayscale
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
        resolution = 0.01  # Set resolution (meters per pixel)
        world_crater_centroids = (crater_centroids) * resolution  # No need for origin correction as (0, 0) is the origin now

        # Print centroid coordinates for large craters
        print("Centroids of Large Craters (World Coordinates):")
        for i, (wx, wy) in enumerate(world_crater_centroids):
            print(f"Crater {i+1}: X = {wx:.3f} m, Y = {wy:.3f} m")
        
        return crater_centroids, resolution
    else:
        print("No large craters found.")
        return None, None

# Function to calculate diameters and visualize craters
def calculate_diameters_and_visualize(image_path, crater_centroids, resolution):
    # Load the black-and-white image
    bw_image = plt.imread(image_path)

    # Ensure grayscale normalization (some PGM files are in [0, 255] and others in [0, 1])
    if bw_image.max() > 1:
        bw_image = bw_image / 255.0  # Normalize to range [0,1]

    # Invert the image vertically (flip it upside down)
    bw_image = np.flipud(bw_image)

    # Threshold to identify black regions (craters)
    binary_map = bw_image < 0.5  # Consider pixels below 0.5 intensity as "black"

    # Create a copy of the image for visualization
    output_image = np.copy(bw_image)

    # Convert the grayscale image to RGB for visualization with colored circles
    output_image = cv2.cvtColor((output_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Function to find the radius by moving outward from the centroid
    def find_radius(binary_map, centroid, max_radius=1000):
        x, y = int(centroid[0]), int(centroid[1])
        height, width = binary_map.shape
        
        # Initialize radius as the maximum possible radius
        radius = max_radius
        
        # Check in all directions (360 degrees)
        angles = np.linspace(0, 2 * np.pi, 360)  # 1 degree increments
        for a in angles:
            for r in range(1, max_radius):
                # Calculate the point on the circle
                cx = int(x + r * np.cos(a))
                cy = int(y + r * np.sin(a))
                
                # Ensure the point is within the image bounds
                if 0 <= cx < width and 0 <= cy < height:
                    if binary_map[cy, cx]:  # If black pixel is found
                        if r < radius:  # Update radius if this is the smallest radius so far
                            radius = r
                        break  # Move to the next angle
        
        return radius  # Return the smallest radius found

    # Draw circles and calculate diameters
    for i, (x, y) in enumerate(crater_centroids):
        # Find the radius for the current centroid
        radius_pixels = find_radius(binary_map, (x, y))
        
        # Convert radius to meters
        radius_meters = radius_pixels * resolution
        
        # Calculate the diameter in meters
        diameter_meters = 2 * radius_meters
        
        # Print the diameter in meters for all craters
        print(f"Crater C{i+1}: Diameter = {diameter_meters:.3f} meters")
        
        # Convert centroid coordinates to meters
        x_meters = x * resolution
        y_meters = y * resolution
        
        # Print centroid coordinates in meters only if diameter is below 0.3 meters
        if diameter_meters < 0.35:
            print(f"  Centroid of Crater C{i+1}: X = {x_meters:.3f} m, Y = {y_meters:.3f} m")
        
        # Draw the circle on the image (in pixels)
        cv2.circle(output_image, (int(x), int(y)), radius_pixels, (0, 255, 0), 2)  # Green circle with thickness 2
        
        # Add text label (C1, C2, ...) near the circle
        cv2.putText(output_image, f"C{i+1}", (int(x) + radius_pixels + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the image with circles
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap="gray", origin="lower")  # Use 'upper' to keep the original orientation
    plt.title("Circles Drawn Around Centroids")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Paths to the images
image_path_1 = "/home/simson/simson_ws/CMU_Capstone_Project/Lunar_ROADSTER_ws/src/mapping/costmap/V2/gradable_craters.pgm"
image_path_2 = "/home/simson/simson_ws/CMU_Capstone_Project/Lunar_ROADSTER_ws/src/mapping/costmap/V2/gradable_craters_diameter.pgm"

# Step 1: Calculate centroids from the first image
crater_centroids, resolution = calculate_centroids(image_path_1)

# Step 2: Calculate diameters and visualize craters on the second image
if crater_centroids is not None:
    calculate_diameters_and_visualize(image_path_2, crater_centroids, resolution)