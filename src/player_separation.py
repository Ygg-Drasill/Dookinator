import math
import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from scipy.spatial import distance


def player_separation(frame, bboxes, box_id, class_id):
    cropped_images = []
    all_color_tuple = []
    bboxes = np.array(bboxes).astype(int)

    for (x_min, y_min, x_max, y_max) in bboxes:
        x_mid = math.floor((x_max + x_min) / 2)
        y_mid = math.floor((y_max + y_min) / 2)

        cropped = frame[y_mid-15:y_mid, x_mid-5:x_mid+5]
        cropped_images.append(cropped)


    for cropped_image in cropped_images:
        color_tuple = k_means(cropped_image)
        all_color_tuple.append(color_tuple)

    id_to_tuple = dict(zip(box_id, all_color_tuple))

    groups = rgb_split(id_to_tuple)
    print(groups)




    return groups

def k_means(image, k = 3):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', image)
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=3, n_init=5)
    kmeans.fit(pixels)

    counter = Counter(kmeans.labels_)
    dominant_color_index = max(counter, key=counter.get)
    dominant_color = kmeans.cluster_centers_[dominant_color_index].astype(int)

    return tuple(dominant_color)


def rgb_split(rgb_values_tuple):
    # Convert NumPy int64 to Python int
    rgb_values = [(int(id_), (int(r), int(g), int(b))) for id_, (r, g, b) in rgb_values_tuple.items()]

    # Extract just the RGB values for clustering
    ids, colors = zip(*rgb_values)  # Unzips into separate lists
    colors_array = np.array(colors)  # Convert to NumPy array

    # Step 1: K-Means Clustering with 2 groups
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(colors_array)  # Assigns each color to a cluster

    # Step 2: Separate into two groups
    group1 = [(ids[i], colors[i]) for i in range(len(ids)) if labels[i] == 0]
    group2 = [(ids[i], colors[i]) for i in range(len(ids)) if labels[i] == 1]

    # Step 3: Balance groups (if needed)
    while abs(len(group1) - len(group2)) > 1:
        # Move the closest color from the bigger group to the smaller one
        if len(group1) > len(group2):
            big_group, small_group = group1, group2
        else:
            big_group, small_group = group2, group1

        # Find the point in the bigger group closest to the smaller group's centroid
        small_group_colors = np.array([color for _, color in small_group])
        big_group_colors = np.array([color for _, color in big_group])
        small_centroid = np.mean(small_group_colors, axis=0)

        distances = [distance.euclidean(color, small_centroid) for _, color in big_group]
        min_index = np.argmin(distances)

        # Move the closest point
        small_group.append(big_group.pop(min_index))

    # Step 4: Create ID-to-Group Mapping
    id_to_group = {}
    for id_, _ in group1:
        id_to_group[id_] = 0  # Group 0
    for id_, _ in group2:
        id_to_group[id_] = 1  # Group 1

    return id_to_group
