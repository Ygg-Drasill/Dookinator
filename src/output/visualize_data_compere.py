import json
import os
import sys

import cv2
import numpy as np
import yaml

from src.definitions import CONFIG_PATH, ROOT_DIR
from src.detectionframe.game_detection_frame import GameDetectionFrame
from src.output.common import read_football_output

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    output_jsonl_file_path = config['output_jsonl_relative_file_path']
    real_data_jsonl_file_path = config['match']['tracking_produced']
    meta_file = config['match']['meta_file']

with open(os.path.join(ROOT_DIR, str(meta_file)), 'r') as f:
    data = json.load(f)

height = 480
width = 640

frame_cap = 10000

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
    our_frames = read_football_output(output_jsonl_file_path)
    real_frames = read_football_output(real_data_jsonl_file_path)

    for i in range(min(len(our_frames), len(real_frames))):
        visualize_data(our_frames[i], real_frames[i])

    cv2.destroyAllWindows()
    return 0


def visualize_data(our_data: GameDetectionFrame, real_data: GameDetectionFrame):
    image = gen_image(our_data, real_data)
    cv2.imshow("Visualization of data", image)
    cv2.waitKey(1)

    pass

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

        real_width = (np.take(center_of_circle, 0) + football_field_width / 2) * width / football_field_width
        real_height = (np.take(center_of_circle, 1) + football_field_height / 2) * height / football_field_height

        new_image = cv2.circle(image, (int(real_width), int(real_height)), radius, color, thickness)
        pass

    return new_image

if __name__ == '__main__':
    sys.exit(main())

