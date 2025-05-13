import numpy as np
import supervision as sv
from src.fieldPitch.KeyPointSaver import KeyPointSaver


class DummyKeyPoints(sv.KeyPoints):
    def __init__(self, xy, class_id=None, confidence=None, data=None):
        self.xy = xy
        self.class_id = class_id if class_id is not None else np.array([0])
        self.confidence = confidence if confidence is not None else np.array([1.0])
        self.data = data if data is not None else np.array([[]])

    def __len__(self):
        return len(self.xy)

    def __eq__(self, other):
        if not isinstance(other, DummyKeyPoints):
            return False
        return np.array_equal(self.xy, other.xy) and \
            np.array_equal(self.class_id, other.class_id) and \
            np.array_equal(self.confidence, other.confidence) and \
            np.array_equal(self.data, other.data)


def test_save_and_load_good_key_points():
    saver = KeyPointSaver()

    xy = np.array([[1, 2], [3, 4]])
    key_points = DummyKeyPoints(xy=xy, class_id=np.array([1, 2]), confidence=np.array([0.9, 0.8]),
                                data=np.array([[10], [20]]))

    saver.save_good_key_points(xy, key_points)
    loaded = saver.load_good_key_points()

    assert len(loaded) == 2
    assert np.array_equal(loaded[0], xy)
    assert loaded[1] == key_points


def test_load_good_key_points_no_data():
    saver = KeyPointSaver()
    assert saver.load_good_key_points() == []


def test_load_good_key_points_empty_xy():
    saver = KeyPointSaver()
    xy = np.array([])
    key_points = DummyKeyPoints(xy=xy, class_id=np.array([]), confidence=np.array([]), data=np.array([[]]))
    saver.save_good_key_points(xy, key_points)
    assert saver.load_good_key_points() == []


def load_good_key_points(self):
    if (self.last_good_xy is not None and len(self.last_good_xy) > 0) and \
            (self.last_good_key_points is not None and len(self.last_good_key_points.xy) > 0):
        return [self.last_good_xy, self.last_good_key_points]
    return []