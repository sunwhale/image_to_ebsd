from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
from scipy import stats
import os


def get_files(path):
    fns = []
    for root, dirs, files in os.walk(path):
        for fn in files:
            fns.append([root, fn])
    return fns


def sub_dirs(path):
    sub_dirs = next(os.walk(path))[1]
    return sub_dirs


def files_in_dir(path):
    file_list = []
    for filename in sorted(next(os.walk(path))[2]):
        file_list.append(filename)
    return file_list


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
        pixels[cluster_mask] = np.round(cluster_centers[i])

    # Reshape the pixels back to the original image shape
    new_image_data = pixels.reshape((height, width, 3))

    new_image = Image.fromarray(new_image_data.astype(np.uint8))
    return new_image


def find_similar_regions_with_edges(image, num_clusters=5, radius=1):
    image_data = np.array(image)
    height, width, _ = image_data.shape

    # Reshape the image data to a 2D array of pixels
    pixels = image_data.reshape((-1, 3))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Reshape the labels back to the original image shape
    labels = labels.reshape((height, width))

    # Use Canny edge detection to find edges
    edges = cv2.Canny(image_data, 100, 200)

    # Iterate through each edge pixel
    for i in range(height):
        for j in range(width):
            if edges[i, j] != 0:
                # Get neighboring pixel colors within radius
                neighbors = get_neighbors(labels, i, j, radius)
                if neighbors:
                    # Calculate mode color of neighbors
                    mode_color = stats.mode(neighbors).mode
                    # Set edge pixel color to mode color
                    labels[i, j] = mode_color

    # Reshape the labels back to a 1D array
    new_labels = labels.reshape((-1,))
    for i in range(num_clusters):
        cluster_mask = (new_labels == i)
        pixels[cluster_mask] = np.round(cluster_centers[i])

    # Reshape the pixels back to the original image shape
    new_image_data = pixels.reshape((height, width, 3))
    new_image = Image.fromarray(new_image_data.astype(np.uint8))
    return new_image


def get_neighbors(labels, i, j, radius=1):
    neighbors = []
    height, width = labels.shape
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width:
                neighbors.append(labels[ni, nj])
    return neighbors


if __name__ == '__main__':
    base_path = r'F:\GitHub\image_to_ebsd\imgs'
    for sub_path in sub_dirs(base_path):
        print(sub_path)
        for filename in files_in_dir(os.path.join(base_path, sub_path)):
            if filename.split('.')[-1] == 'jpg':
                image_name = os.path.join(base_path, sub_path, filename)
                input_image = Image.open(image_name)
                image_with_kmeans = find_similar_regions(input_image, num_clusters=12)
                output_image = find_similar_regions_with_edges(image_with_kmeans, num_clusters=12, radius=5)
                output_image.save(image_name)
