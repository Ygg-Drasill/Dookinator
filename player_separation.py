import cv2
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans


def player_separation(frame, bboxes):
    cropped_images = []
    player_color = []
    bboxes = np.array(bboxes).astype(int)

    for (x_min, y_min, x_max, y_max) in bboxes:
        cropped = frame[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped)

    for cropped_image in cropped_images:
        color_tuple = k_means(cropped_image)
        color_name = closest_color(color_tuple)
        player_color.append(color_name)


    return player_color

def k_means(image, k = 3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #image to a list of pixels
    pixels = image.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(pixels)

    # Get the most common color
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
        "gray": (128, 128, 128),
        "light gray": (211, 211, 211),
        "dark gray": (169, 169, 169),
        "red": (255, 0, 0),
        "dark red": (139, 0, 0),
        "light red": (255, 102, 102),
        "orange": (255, 165, 0),
        "dark orange": (255, 140, 0),
        "light orange": (255, 200, 87),
        "yellow": (255, 255, 0),
        "gold": (255, 215, 0),
        "light yellow": (255, 255, 153),
        "green": (0, 255, 0),
        "dark green": (0, 100, 0),
        "light green": (144, 238, 144),
        "lime": (0, 255, 0),
        "olive": (128, 128, 0),
        "cyan": (0, 255, 255),
        "teal": (0, 128, 128),
        "turquoise": (64, 224, 208),
        "blue": (0, 0, 255),
        "dark blue": (0, 0, 139),
        "light blue": (173, 216, 230),
        "sky blue": (135, 206, 235),
        "navy": (0, 0, 128),
        "purple": (128, 0, 128),
        "dark purple": (75, 0, 130),
        "violet": (238, 130, 238),
        "indigo": (75, 0, 130),
        "pink": (255, 20, 147),
        "hot pink": (255, 105, 180),
        "light pink": (255, 182, 193),
        "brown": (165, 42, 42),
        "dark brown": (101, 67, 33),
        "beige": (245, 245, 220),
        "tan": (210, 180, 140),
        "maroon": (128, 0, 0),
        "salmon": (250, 128, 114),
        "coral": (255, 127, 80),
        "peach": (255, 218, 185),
        "lavender": (230, 230, 250),
        "magenta": (255, 0, 255)
    }

    for name, rgb in color_dict.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(rgb_tuple))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name

    return closest_color_name
