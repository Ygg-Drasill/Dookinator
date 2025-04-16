from enum import Enum
import numpy as np

from src.detectionframe.ball_detection_frame import BallDetectionFrame

class LastTouch(Enum):
    Home = 1
    Away = 2

class GameDetectionFrame:
    def __init__(self, period: int, frame_idx: int, game_clock: float, wall_clock: int, home_players: np.ndarray, away_players: np.ndarray, ball: BallDetectionFrame, live: bool, last_touch: LastTouch):
        self.period = period
        self.frame_idx = frame_idx
        self.game_clock = game_clock
        self.wall_clock = wall_clock
        self.home_players = home_players
        self.away_players = away_players
        self.ball = ball
        self.live = live
        self.last_touch = last_touch
        pass

    def to_dict(self):
        return {
            "period": self.period,
            "frameIdx": self.frame_idx,
            "gameClock": self.game_clock,
            "wallClock": self.wall_clock,
            "homePlayers": [player.to_dict() for player in self.home_players],
            "awayPlayers": [player.to_dict() for player in self.away_players],
            "ball": self.ball.to_dict() if self.ball else None,
            "live": self.live,
            "lastTouch": self.last_touch.name
        }
