import cv2
import numpy as np
import yaml

from src.definitions import CONFIG_PATH
from src.detectionframe.game_detection_frame import GameDetectionFrame

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    output_jsonl_file_path = config['output_jsonl_relative_file_path']
    real_data_jsonl_file_path = config['real_data_jsonl_relative_file_path']


height = 480
width = 640

football_field_width = 1920
football_field_height = 1080

#shape settings
our_away_color = (255, 255, 0)
our_home_color = (255, 0, 255)
real_away_color = (0, 255, 0)
real_home_color = (0, 0, 255)
radius = 2
thickness = 1

blank_image = np.zeros((height,width,3), np.uint8)

def visualize_data(frame: int):
    our_data = read_football_output(output_jsonl_file_path)
    real_data = read_football_output(real_data_jsonl_file_path)

    image = gen_image(our_data, real_data)

    cv2.imshow("visualization of data", image)
    pass

def read_football_output(file_path: str) -> GameDetectionFrame:

    



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

        real_width = np.take(center_of_circle, 0) / football_field_width * width
        real_height = np.take(center_of_circle, 1) / football_field_height * height

        new_image = cv2.circle(image, (int(real_width), int(real_height)), radius, color, thickness)
        pass

    return new_image

