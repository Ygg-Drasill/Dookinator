import json
import os
import sys

import cv2
import numpy as np
import yaml
import pandas as pd

from src.definitions import CONFIG_PATH, ROOT_DIR
from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.detectionframe.player_detection_frame import PlayerDetectionFrame

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    output_jsonl_file_path = config['output_jsonl_relative_file_path']
    real_data_jsonl_file_path = config['real_data_jsonl_relative_file_path']
    meta_file = config['match']['meta_file']

with open(os.path.join(ROOT_DIR, str(meta_file)), 'r') as f:
    data = json.load(f)

height = 480
width = 640

football_field_width = data["pitchLength"]
football_field_height = data["pitchWidth"]

# shape settings
our_away_color = (255, 255, 0)
our_home_color = (255, 0, 255)
real_away_color = (0, 255, 0)
real_home_color = (0, 0, 255)
radius = 2
thickness = 1

blank_image = np.zeros((height, width, 3), np.uint8)

def main() -> int:



    return


def visualize_data(frame: int):
    our_data = read_football_output(output_jsonl_file_path)
    real_data = read_football_output(real_data_jsonl_file_path)

    image = gen_image(our_data, real_data)

    cv2.imshow("visualization of data", image)
    pass

def read_football_output(file_path: str) -> GameDetectionFrame:

    json_obj = pd.read_json(path_or_buf=file_path, lines=True)
    frame = json_obj.iloc[0]

    home_players_raw = frame["home_players"]

    home_players = np.array([
        PlayerDetectionFrame(
            player_id=p["player_id"],
            number=p["number"],
            xyz=np.array(p["xyz"]),
            speed=p["speed"],
            opta_id=p["opta_id"]
        )
        for p in home_players_raw
    ])

    away_players_raw = frame["away_players"]

    away_players = np.array([
        PlayerDetectionFrame(
            player_id=p["player_id"],
            number=p["number"],
            xyz=np.array(p["xyz"]),
            speed=p["speed"],
            opta_id=p["opta_id"]
        )
        for p in away_players_raw
    ])

    ball_raw = frame["ball"]

    if ball_raw is not None:
        ball = BallDetectionFrame(
            xyz=np.array(ball_raw["xyz"]),
            speed=ball_raw["speed"]
        )
    else:
        ball = None

    game_detection_frame = GameDetectionFrame(period=0, frame_idx=frame, game_clock=0, wall_clock=0, home_players=home_players, away_players=away_players, ball=ball, live=True, last_touch=LastTouch.Home)
    return game_detection_frame

def gen_image(our_game_detection_frame: GameDetectionFrame, real_game_detection_frame: GameDetectionFrame) -> np.array:
    image = blank_image.copy()

    image = read_players(our_game_detection_frame.away_players, image, our_away_color)
    image = read_players(our_game_detection_frame.home_players, image, our_home_color)
    image = read_players(real_game_detection_frame.away_players, image, real_away_color)
    image = read_players(real_game_detection_frame.home_players, image, real_home_color)

    return image

def read_players(players: np.ndarray, image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:

    new_image = image

    for player in players:
        center_of_circle = player.xyz[:2]

        real_width = np.take(center_of_circle, 0) / football_field_width * width
        real_height = np.take(center_of_circle, 1) / football_field_height * height

        new_image = cv2.circle(image, (int(real_width), int(real_height)), radius, color, thickness)
        pass

    return new_image

if __name__ == '__main__':
    sys.exit(main())