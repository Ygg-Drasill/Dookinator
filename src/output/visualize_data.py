import cv2
import numpy as np

from src.detectionframe.game_detection_frame import GameDetectionFrame

height = 480
width = 640

radius = 2
away_color = (0, 255, 0)
thickness = 1

blank_image = np.zeros((height,width,3), np.uint8)

def visualize_data(game_detection_frame: GameDetectionFrame):
    image = blank_image.copy()
    for away_player in game_detection_frame.away_players:
        #centerOfCircle = away_player.

        image = cv2.circle(image, centerOfCircle, radius, away_color, thickness)

        pass



    cv2.imshow("visualization of data", )
    pass

def read_football_output():


    pass

def gen_image():


    pass


