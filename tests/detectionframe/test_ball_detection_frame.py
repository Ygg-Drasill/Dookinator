import unittest
import numpy as np
from src.detectionframe.ball_detection_frame import BallDetectionFrame


class TestBallDetectionFrame(unittest.TestCase):
    def test_initialization(self):
        xyz = np.array([1.0, 2.0, 3.0])
        speed = 5.5
        frame = BallDetectionFrame(xyz, speed)

        np.testing.assert_array_equal(frame.xyz, xyz)
        self.assertEqual(frame.speed, speed)

    def test_to_dict(self):
        xyz = np.array([1.0, 2.0, 3.0])
        speed = 5.5
        frame = BallDetectionFrame(xyz, speed)

        expected_dict = {"xyz": [1.0, 2.0, 3.0], "speed": 5.5}
        self.assertEqual(frame.to_dict(), expected_dict)

    def test_zero_values(self):
        xyz = np.array([0.0, 0.0, 0.0])
        speed = 0.0
        frame = BallDetectionFrame(xyz, speed)

        expected_dict = {"xyz": [0.0, 0.0, 0.0], "speed": 0.0}
        self.assertEqual(frame.to_dict(), expected_dict)

    def test_negative_values(self):
        xyz = np.array([-1.0, -2.0, -3.0])
        speed = -4.5
        frame = BallDetectionFrame(xyz, speed)

        expected_dict = {"xyz": [-1.0, -2.0, -3.0], "speed": -4.5}
        self.assertEqual(frame.to_dict(), expected_dict)

    def test_large_values(self):
        xyz = np.array([1e6, 2e6, 3e6])
        speed = 1e3
        frame = BallDetectionFrame(xyz, speed)

        expected_dict = {"xyz": [1e6, 2e6, 3e6], "speed": 1e3}
        self.assertEqual(frame.to_dict(), expected_dict)