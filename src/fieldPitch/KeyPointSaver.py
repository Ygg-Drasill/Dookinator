import numpy as np
import supervision as sv

class KeyPointSaver:

    def __init__(self):
        self.last_good_xy = None
        self.last_good_key_points = None

    def save_good_key_points(self, xy: np.ndarray, key_points: sv.KeyPoints):
        self.last_good_xy = xy
        self.last_good_key_points = key_points
        pass

    def load_good_key_points(self):
        if self.last_good_xy is not None and self.last_good_key_points is not None:
            if len(self.last_good_xy) > 0 and len(self.last_good_key_points) > 0:
                return [self.last_good_xy, self.last_good_key_points]
        return []
