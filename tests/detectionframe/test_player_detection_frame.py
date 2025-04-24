import unittest
import numpy as np
from src.detectionframe.player_detection_frame import PlayerDetectionFrame

class TestPlayerDetectionFrame(unittest.TestCase):

    def test_initialization(self):
        player_id = "test_id"
        number = 0
        xyz = np.array([1.0, 2.0, 3.0])
        speed = 5.5
        opta_id = 0
        frame = PlayerDetectionFrame(player_id, number, xyz, speed, opta_id)

        self.assertEqual(frame.player_id, player_id)
        self.assertEqual(frame.number, number)
        np.testing.assert_array_equal(frame.xyz, xyz)
        self.assertEqual(frame.speed, speed)
        self.assertEqual(frame.opta_id, opta_id)
        pass

    def test_to_dict(self):
        player_id = "testId"
        number = 0
        xyz = np.array([1.0, 2.0, 3.0])
        speed = 5.5
        opta_id = 0
        frame = PlayerDetectionFrame(player_id, number, xyz, speed, opta_id)

        expected_dict = {"playerId": "testId", "number": 0, "xyz": [1.0, 2.0, 3.0], "speed": 5.5, "optaId": 0}
        self.assertEqual(frame.to_dict(), expected_dict)


        pass

    pass
