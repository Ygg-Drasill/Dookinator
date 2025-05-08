import cv2
import numpy as np

from src.fieldPitch.view import ViewTransformer


class KeyPointsManager:
    def __init__(self, max_length):
        self.key_points_list = []
        self.max_length = max_length

    def add_new_key_point(self, new_key_points):
        if len(self.key_points_list) >= self.max_length:
            self.key_points_list.pop(0)
        self.key_points_list.append(new_key_points)

    def get_transformer(self, soccer_field_config):

        key_points_arrays = [kp.astype(np.float32) for kp in self.key_points_list]

        stacked = np.stack(key_points_arrays)

        average_key_points = np.mean(stacked, axis=0)

        mask = (
                (average_key_points[:, 0] > 1)
                & (average_key_points[:, 1] > 1)
        )

        transformer = ViewTransformer(
            source=average_key_points[mask].astype(np.float32),
            target=np.array(soccer_field_config.vertices)[mask].astype(np.float32)
        )

        return transformer