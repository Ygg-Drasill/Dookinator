import numpy as np
import supervision as sv

class KeyPointSaver:
    def save_good_key_points(self, xy: np.ndarray, key_points: sv.KeyPoints):
        self.last_good_xy = xy
        self.last_good_key_points = key_points
        pass

    def load_good_key_points(self):
        if self.last_good_xy and self.last_good_key_points:
            return [self.last_good_xy, self.last_good_key_points]
        else:
            return []
