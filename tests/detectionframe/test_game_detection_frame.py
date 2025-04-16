import unittest
import numpy as np

from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch


class TestGameDetectionFrame(unittest.TestCase):

    def test_initialization(self):

        period = 0
        frame_idx = 0
        game_clock = 0
        wall_clock = 0
        home_players = np.array([])
        away_players = np.array([])
        ball = BallDetectionFrame(np.array([0, 0, 0]), 0)
        live = True
        last_touch = LastTouch.Home

        frame = GameDetectionFrame(period, frame_idx, game_clock, wall_clock, home_players, away_players, ball, live, last_touch)

        self.assertEqual(frame.period, period)
        self.assertEqual(frame.frame_idx, frame_idx)
        self.assertEqual(frame.game_clock, game_clock)
        self.assertEqual(frame.wall_clock, wall_clock)
        np.testing.assert_array_equal(frame.home_players, home_players)
        np.testing.assert_array_equal(frame.away_players, away_players)
        self.assertEqual(frame.ball, ball)
        self.assertEqual(frame.live, live)
        self.assertEqual(frame.last_touch, last_touch)

        pass

    def test_to_dict(self):

        period = 0
        frame_idx = 0
        game_clock = 0
        wall_clock = 0
        home_players = np.array([])
        away_players = np.array([])
        ball = BallDetectionFrame(np.array([0, 0, 0]), 0)
        live = True
        last_touch = LastTouch.Home

        frame = GameDetectionFrame(period, frame_idx, game_clock, wall_clock, home_players, away_players, ball, live,
                                   last_touch)

        expected_dict = {"period": 0,"frame_idx": 0, "game_clock": 0, "wall_clock": 0, "home_players": [player.to_dict() for player in np.array([])], "away_players": [player.to_dict() for player in np.array([])], "ball": BallDetectionFrame(np.array([0, 0, 0]), 0).to_dict(), "live": True, "last_touch": LastTouch.Home.name}
        self.assertEqual(expected_dict, frame.to_dict())

        pass


    pass