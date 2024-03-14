from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


def find_similar_regions(image, num_clusters=5):
    image_data = np.array(image)
    height, width, _ = image_data.shape

    # Reshape the image data to a 2D array of pixels
    pixels = image_data.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Assign average color to each region
    for i in range(num_clusters):
        cluster_mask = (labels == i)
        pixels[cluster_mask] = cluster_centers[i]

    # Reshape the pixels back to the original image shape
    new_image_data = pixels.reshape((height, width, 3))

    new_image = Image.fromarray(new_image_data.astype(np.uint8))
    return new_image


# Open an image file
input_image = Image.open("./imgs/3/3.jpg")

# Find and replace similar color regions
output_image = find_similar_regions(input_image, num_clusters=12)

# Save the output image
output_image.save("output_image.jpg")
