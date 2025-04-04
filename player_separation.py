import math

import cv2
import numpy as np
from collections import Counter

from sklearn.cluster import KMeans


def player_separation(frame, bboxes, box_id, class_id):
    cropped_images = []
    player_color = []
    bboxes = np.array(bboxes).astype(int)

    for (x_min, y_min, x_max, y_max) in bboxes:
        x_mid = math.floor((x_max + x_min) / 2)
        y_mid = math.floor((y_max + y_min) / 2)

        cropped = frame[y_mid-14:y_mid, x_mid-7:x_mid+7]
        cropped_images.append(cropped)


    for cropped_image in cropped_images:
        color_tuple = k_means(cropped_image)
        color_name = closest_color(color_tuple)
        player_color.append(color_name)
    print(player_color)
    print(box_id)
    id_to_color = dict(zip(box_id, player_color))
    return id_to_color

def k_means(image, k = 3):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', image)
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=3, n_init=10)
    kmeans.fit(pixels)

    counter = Counter(kmeans.labels_)
    dominant_color_index = max(counter, key=counter.get)
    dominant_color = kmeans.cluster_centers_[dominant_color_index].astype(int)

    return tuple(dominant_color)


def closest_color(rgb_tuple):
    min_distance = float("inf")
    closest_color_name = None
    color_dict = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "orange": (255, 165, 0),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "cyan": (0, 255, 255),
        "blue": (0, 0, 255),
        "purple": (128, 0, 128),
        "pink": (255, 20, 147),
        "brown": (165, 42, 42),
        "magenta": (255, 0, 255)
    }

    for name, rgb in color_dict.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(rgb_tuple))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name

    return closest_color_name
