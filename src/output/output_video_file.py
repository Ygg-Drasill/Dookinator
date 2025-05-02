import json
import os

import cv2
import yaml

from src.definitions import CONFIG_PATH, ROOT_DIR

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    match = config['match']

with open(os.path.join(ROOT_DIR, str(match['meta_file'])), 'r') as f:
        frame_data = json.load(f)

class SoccerVideoWriter:
    def __init__(self):
        frame_width = 1920
        frame_height = 1080
        fps = int(frame_data['fps'])
        output_path = match['video_output']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def __del__(self):
        self.out.release()

    def write_frame(self, frame):
        self.out.write(frame)