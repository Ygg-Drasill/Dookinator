import json
import os
import numpy as np
import yaml
import supervision as sv
from numpy.f2py.auxfuncs import throw_error

from src.definitions import CONFIG_PATH, ROOT_DIR
from src.fieldPitch.KeyPointSaver import KeyPointSaver
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration
from src.fieldPitch.view import ViewTransformer

key_point_saver = KeyPointSaver()

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

    data = load_pitch_data()

    pitch_length = data["pitchLength"]
    pitch_width = data["pitchWidth"]

    soccer_field_config = SoccerPitchConfiguration(width=pitch_width, length=pitch_length)

    mask = (
            (key_points.xy[0][:, 0] > 1)
            & (key_points.xy[0][:, 1] > 1)
            & (key_points.confidence[0] > 0.5)
    )

    #check if there are 3 or fewer trues
    if mask.sum() <= 3:
        good_values = key_point_saver.load_good_key_points()
        if len(good_values) > 0:
            xy = good_values[0]
            key_points = good_values[1]

            mask = (
                    (key_points.xy[0][:, 0] > 1)
                    & (key_points.xy[0][:, 1] > 1)
                    & (key_points.confidence[0] > 0.5)
            )
        else:
            raise ValueError('No good key points found for given frame')
    else:
        key_point_saver.save_good_key_points(xy, key_points)

    transformer = ViewTransformer(
        source=key_points.xy[0][mask].astype(np.float32),
        target=np.array(soccer_field_config.vertices)[mask].astype(np.float32)
    )

    xy = np.float32(xy)

    transformed_xy = transformer.transform_points(point=xy)[0][0]

    transformed_xy[0] = transformed_xy[0] - (pitch_length / 2)
    transformed_xy[1] = transformed_xy[1] - (pitch_width / 2)

    return transformed_xy