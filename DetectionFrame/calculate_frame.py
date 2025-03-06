import numpy as np
import yaml
from supervision import Detections

from DetectionFrame.ball_detection_frame import BallDetectionFrame
from DetectionFrame.DetectionsVariablesCalculator.detection_location import get_screen_location_of_bounding_box
from DetectionFrame.game_detection_frame import GameDetectionFrame, LastTouch
from DetectionFrame.player_detection_frame import PlayerDetectionFrame

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    selected_class_ids = config['yolo']['selected_class_ids']
    frame_rate = config['byte_track']['frame_rate']

def calculate_detection_frame(detections: Detections, frame_idx: int) -> GameDetectionFrame:

    game_clock = frame_idx * (1 / frame_rate)

    # just set for testing
    live = True
    period = 1
    wall_clock = 0
    last_touch = LastTouch.Home

    home_player_detections = []
    away_player_detections = []

    ball_detection = None

    for i, class_id in enumerate(detections.class_id):

        if class_id == selected_class_ids['ball']['id']:
            # location is just screen location for now
            location = get_screen_location_of_bounding_box(detections.xyxy[i])

            # just set for testing
            speed = 0

            ball_detection = BallDetectionFrame(location, speed)
        if class_id == selected_class_ids['player']['id'] or class_id == selected_class_ids['goalkeeper']['id']:

            # location is just screen location for now
            location = get_screen_location_of_bounding_box(detections.xyxy[i])

            # just set for testing
            speed = 0
            player_id = "0"
            number = 0
            opta_id = 0

            player = PlayerDetectionFrame(player_id, number, location, speed, opta_id)
            home_player_detections.append(player)

        pass

    # spilt up in teams


    return GameDetectionFrame(period, frame_idx, game_clock, wall_clock, np.array(home_player_detections), np.array(away_player_detections), ball_detection, live, last_touch)