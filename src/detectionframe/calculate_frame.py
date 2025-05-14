import numpy as np
import yaml
from supervision import Detections, KeyPoints

from src.definitions import CONFIG_PATH
from src.detectionframe.ball_detection_frame import BallDetectionFrame
from src.detectionframe.detectionsVariablesCalculator.detection_location import get_location_of_bounding_box
from src.detectionframe.game_detection_frame import GameDetectionFrame, LastTouch
from src.detectionframe.player_detection_frame import PlayerDetectionFrame

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    selected_class_ids = config['yolo']['selected_class_ids']
    frame_rate = config['byte_track']['frame_rate']

def calculate_detection_frame(detections: Detections, frame_idx: int, keypoints: KeyPoints) -> GameDetectionFrame:

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

            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            midpoint = np.array([(x1 + x2) / 2, y2])

            location = get_location_of_bounding_box(midpoint, keypoints)

            speed = 0

            ball_detection = BallDetectionFrame(np.array(location), speed)
        if class_id == selected_class_ids['player']['id'] or class_id == selected_class_ids['goalkeeper']['id']:

            bbox = detections.xyxy[i]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            midpoint = np.array([(x1 + x2) / 2, y2])

            location = get_location_of_bounding_box(midpoint, keypoints)

            speed = 0
            player_id = "0"
            number = 0
            opta_id = 0

            player = PlayerDetectionFrame(player_id, number, location, speed, opta_id)

            if detections["player_team"][i] == 1:
                away_player_detections.append(player)
            if detections["player_team"][i] == 0:
                home_player_detections.append(player)
        pass

    return GameDetectionFrame(period, frame_idx, game_clock, wall_clock, np.array(home_player_detections), np.array(away_player_detections), ball_detection, live, last_touch)