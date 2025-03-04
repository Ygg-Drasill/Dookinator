from enum import Enum
from typing import List

from detectionFrame.ball_detection_frame import BallDetectionFrame
from detectionFrame.player_detection_frame import PlayerDetectionFrame

class LastTouch(Enum):
    Home = 1
    Away = 2

class GameDetectionFrame:
    def __init__(self, period: int, frame_idx: int, game_clock: float, wall_clock: int, home_players: List[PlayerDetectionFrame], away_players: List[PlayerDetectionFrame], ball: BallDetectionFrame, live: bool, last_touch: LastTouch):
        self.period = period
        self.frame_idx = frame_idx
        self.game_clock = game_clock
        self.wall_clock = wall_clock
        self.home_players = home_players
        self.away_players = away_players
        self.ball = ball
        pass

    def to_dict(self):
        return {
            "period": self.period,
            "frame_idx": self.frame_idx,
            "game_clock": self.game_clock,
            "wall_clock": self.wall_clock,
            "home_players": [player.to_dict() for player in self.home_players],
            "away_players": [player.to_dict() for player in self.away_players],
            "ball": self.ball.to_dict() if self.ball else None
        }
