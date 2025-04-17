import os

import numpy as np
import pandas as pd

from src.definitions import ROOT_DIR
from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.detectionframe.player_detection_frame import PlayerDetectionFrame


def read_football_output(file_path: str) -> list[GameDetectionFrame]:
    json_iter = pd.read_json(path_or_buf=os.path.join(ROOT_DIR, file_path), lines=True, chunksize=10000)
    game_detection_frames = []

    json_obj = next(json_iter)

    for _, frame in json_obj.iterrows():
        home_players = np.array([
            PlayerDetectionFrame(
                player_id=p["playerId"],
                number=p["number"],
                xyz=np.array(p["xyz"]),
                speed=p["speed"],
                opta_id=p["optaId"]
            ) for p in frame["homePlayers"]
        ])

        away_players = np.array([
            PlayerDetectionFrame(
                player_id=p["playerId"],
                number=p["number"],
                xyz=np.array(p["xyz"]),
                speed=p["speed"],
                opta_id=p["optaId"]
            ) for p in frame["awayPlayers"]
        ])

        ball_raw = frame["ball"]
        ball = BallDetectionFrame(xyz=np.array(ball_raw["xyz"]), speed=ball_raw["speed"]) if ball_raw and not pd.isna(ball_raw) else None

        game_detection_frame = GameDetectionFrame(
            period=0, frame_idx=frame["frameIdx"], game_clock=0, wall_clock=0,
            home_players=home_players, away_players=away_players,
            ball=ball, live=True, last_touch=LastTouch.Home
        )
        game_detection_frames.append(game_detection_frame)

    return game_detection_frames