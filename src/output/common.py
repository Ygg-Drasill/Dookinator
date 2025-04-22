import os

import cv2
import numpy as np
import pandas as pd

from src.definitions import ROOT_DIR
from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.detectionframe.player_detection_frame import PlayerDetectionFrame
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration


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

def draw_soccer_field(config: SoccerPitchConfiguration, padding: int = 50, line_thickness: int = 4, scale: float = 10) -> np.ndarray:
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    point_radius = 6

    soccer_field_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    )

    soccer_field_image[:] = [0, 128, 0]

    for start, end in config.edges:
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)

        cv2.line(
            img=soccer_field_image,
            pt1=point1,
            pt2=point2,
            color=[255, 255, 255],
            thickness=line_thickness
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=soccer_field_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=[255, 255, 255],
        thickness=line_thickness
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=soccer_field_image,
            center=spot,
            radius=point_radius,
            color=[255, 255, 255],
            thickness=-1
        )

    return soccer_field_image

def read_players(config: SoccerPitchConfiguration, players: np.ndarray, image: np.ndarray, color: tuple[int, int, int], padding: int = 50,scale: float = 10, pitch: np.ndarray = None) -> np.ndarray:

    new_pitch = pitch
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    for player in players:
        center_of_circle = player.xyz[:2]

        real_width = (np.take(center_of_circle, 0) * scale) + padding + scaled_width / 2
        real_height = (np.take(center_of_circle, 1) * scale) + padding + scaled_length / 2

        new_pitch = cv2.circle(image, (int(real_width), int(real_height)), 4, color, -1)
        pass

    return new_pitch