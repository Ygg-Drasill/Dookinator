import json
import os

import numpy as np
import supervision as sv
import yaml

from src.definitions import CONFIG_PATH, ROOT_DIR
from src.fieldPitch.KeyPointSaver import KeyPointSaver
from src.fieldPitch.KeyPointsManager import KeyPointsManager
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration

key_point_saver = KeyPointSaver()
key_points_manager = KeyPointsManager(10)
no_keypoints_detected = 0
counter = 0

def load_pitch_data():
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
        meta_file = config['match']['meta_file']

    with open(os.path.join(ROOT_DIR, str(meta_file)), 'r') as f:
        return json.load(f)


def get_location_of_bounding_box(xy: np.ndarray, key_points: sv.KeyPoints) -> np.float32:
    """
    Computes the screen location of a bounding box given its (x1, y1, x2, y2) coordinates.

    Args:
        xy (np.ndarray): A 1D NumPy array of shape (2) representing (x1, x2).
        key_points (sv.KeyPoints): The keypoints of the soccer field.

    Returns:
        np.ndarray: A 1D NumPy array with the real position of the player.
    """
    global no_keypoints_detected, counter

    if len(key_points.xy) == 0 or key_points.xy[0].size == 0:
        raise ValueError("key_points.xy is empty or improperly structured.")

    data = load_pitch_data()

    pitch_length = data["pitchLength"]
    pitch_width = data["pitchWidth"]

    soccer_field_config = SoccerPitchConfiguration(width=pitch_width, length=pitch_length)

    mask = (
            (key_points.xy[0][:, 0] > 1)
            & (key_points.xy[0][:, 1] > 1)
    )

    counter = counter + 1
    #check if there are 3 or fewer trues
    if mask.sum() <= 3:
        no_keypoints_detected = no_keypoints_detected + 1
        good_values = key_point_saver.load_good_key_points()
        if len(good_values) > 0:
            xy = good_values[0]
            key_points = good_values[1]
        else:
            raise ValueError('No good key points found for given frame')
    else:
        key_point_saver.save_good_key_points(xy, key_points)

    key_points_manager.add_new_key_point(key_points.xy[0].astype(np.float32))

    xy = np.float32(xy)

    transformed_xy = key_points_manager.get_transformer(soccer_field_config).transform_points(point=xy)[0][0]

    transformed_xy[0] = transformed_xy[0] - (pitch_length / 2)
    transformed_xy[1] = (pitch_width / 2) - transformed_xy[1]

    print("The number of keypoints not detected:" + str(no_keypoints_detected) + "counter:" + str(counter))
    return transformed_xy