import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the black-and-white image
image_path = "/home/simson/simson_ws/CMU_Capstone_Project/Lunar_ROADSTER_ws/src/mapping/costmap/V2/gradable_craters_diameter.pgm"  # Update with the correct path
bw_image = plt.imread(image_path)

# Ensure grayscale normalization (some PGM files are in [0, 255] and others in [0, 1])
if bw_image.max() > 1:
    bw_image = bw_image / 255.0  # Normalize to range [0,1]

# Invert the image vertically (flip it upside down)
bw_image = np.flipud(bw_image)

# Threshold to identify black regions (craters)
binary_map = bw_image < 0.5  # Consider pixels below 0.5 intensity as "black"

# Centroid coordinates obtained from the previous script
# Replace these with your actual centroid coordinates
crater_centroids = np.array([
    [1029.8, 336.0],  # Example centroid 1 (x, y)
    [772.6, 485.7],  # Example centroid 2 (x, y)
    [1135.9, 507.0],  # Example centroid 3 (x, y)
    [952.6, 694.4]   # Example centroid 4 (x, y)
])

# Resolution of the image (meters per pixel)
resolution = 0.01  # Example: 1 pixel = 0.01 meters (update this value as needed)

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

# Create a copy of the image for visualization
output_image = np.copy(bw_image)

# Convert the grayscale image to RGB for visualization with colored circles
output_image = cv2.cvtColor((output_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Get image height for y-axis inversion
image_height = bw_image.shape[0]

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
    if diameter_meters < 0.3:
        print(f"  Centroid of Crater C{i+1}: X = {x_meters:.3f} m, Y = {y_meters:.3f} m")
    
    # Draw the circle on the image (in pixels)
    cv2.circle(output_image, (int(x), int(y)), radius_pixels, (0, 255, 0), 2)  # Green circle with thickness 2
    
    # Add text label (C1, C2, ...) near the circle
    cv2.putText(output_image, f"C{i+1}", (int(x) + radius_pixels + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the image with circles
plt.figure(figsize=(8, 8))
plt.imshow(output_image, cmap="gray", origin="upper")  # Use 'upper' to keep the original orientation
plt.title("Circles Drawn Around Centroids")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()