import cv2
import numpy as np

from src.detectionframe.game_detection_frame import GameDetectionFrame

height = 480
width = 640

football_field_width = 1920
football_field_height = 1080

radius = 2
away_color = (0, 255, 0)
thickness = 1

blank_image = np.zeros((height,width,3), np.uint8)

def visualize_data(game_detection_frame: GameDetectionFrame):
    image = blank_image.copy()
    for home_player in game_detection_frame.home_players:
        center_of_circle = home_player.xyz[:2]

        real_width = np.take(center_of_circle, 0) / football_field_width * width
        real_height = np.take(center_of_circle, 1) / football_field_height * height

        image = cv2.circle(image, (int(real_width), int(real_height)), radius, away_color, thickness)
        pass



    cv2.imshow("visualization of data", image)
    pass

def read_football_output():


    pass

def gen_image():


    pass


