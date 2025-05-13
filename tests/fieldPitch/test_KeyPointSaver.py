import numpy as np
import supervision as sv
from src.fieldPitch.KeyPointSaver import KeyPointSaver

class DummyKeyPoints(sv.KeyPoints):
    def __init__(self, points):
        self.points = points

    def __len__(self):
        return len(self.points)

def test_save_and_load_good_key_points():
    saver = KeyPointSaver()

    xy = np.array([[1, 2], [3, 4]])
    key_points = DummyKeyPoints(points=[[5, 6], [7, 8]])

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
    key_points = DummyKeyPoints(points=[[1, 2]])
    saver.save_good_key_points(xy, key_points)
    assert saver.load_good_key_points() == []

def test_load_good_key_points_empty_keypoints():
    saver = KeyPointSaver()
    xy = np.array([[1, 2]])
    key_points = DummyKeyPoints(points=[])
    saver.save_good_key_points(xy, key_points)
    assert saver.load_good_key_points() == []