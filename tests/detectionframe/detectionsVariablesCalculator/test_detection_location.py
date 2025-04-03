import unittest
import numpy as np
from src.detectionframe.detectionsVariablesCalculator.detection_location import get_screen_location_of_bounding_box

class TestGetScreenLocation(unittest.TestCase):
    def test_valid_bounding_box(self):
        xyxy = np.array([10, 20, 30, 40])
        expected_output = np.array([20, 40])  # Midpoint of x-coordinates (10,30) is 20; bottom y is 40
        np.testing.assert_array_equal(get_screen_location_of_bounding_box(xyxy), expected_output)

    def test_negative_coordinates(self):
        xyxy = np.array([-10, -20, 30, 40])
        expected_output = np.array([10, 40])  # Midpoint of x-coordinates (-10,30) is 10; bottom y is 40
        np.testing.assert_array_equal(get_screen_location_of_bounding_box(xyxy), expected_output)

    def test_zero_bounding_box(self):
        xyxy = np.array([0, 0, 0, 0])
        expected_output = np.array([0, 0])  # Midpoint of x-coordinates (0,0) is 0; bottom y is 0
        np.testing.assert_array_equal(get_screen_location_of_bounding_box(xyxy), expected_output)

    def test_large_values(self):
        xyxy = np.array([1e6, 1e6, 3e6, 4e6])
        expected_output = np.array([2e6, 4e6])  # Midpoint of x-coordinates (1e6,3e6) is 2e6; bottom y is 4e6
        np.testing.assert_array_equal(get_screen_location_of_bounding_box(xyxy), expected_output)


