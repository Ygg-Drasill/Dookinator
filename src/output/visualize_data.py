import json
import os
import sys
import time

import cv2
import yaml

from src.definitions import CONFIG_PATH, ROOT_DIR
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration
from src.output.common import read_football_output, draw_soccer_field, read_players

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    output_jsonl_file_path = config['output_jsonl_relative_file_path']
    meta_file = config['match']['meta_file']


def load_pitch_data():
    with open(os.path.join(ROOT_DIR, str(meta_file)), 'r') as f:
        return json.load(f)

def main() -> int:
    our_frames = read_football_output(output_jsonl_file_path)

    data = load_pitch_data()

    pitch_length = data["pitchLength"]
    pitch_width = data["pitchWidth"]

    soccer_field_config = SoccerPitchConfiguration(width=pitch_width, length=pitch_length)

    for i in range(len(our_frames)):
        image = draw_soccer_field(soccer_field_config, 50, 4, 10)

        home_players = our_frames[i].home_players

        image = read_players(soccer_field_config, image, home_players,  (0, 0, 255), 50, 10)

        cv2.imshow("Visualization of data", image)
        cv2.waitKey(1)

        time.sleep(1 / 25)
    cv2.destroyAllWindows()
    return 0






if __name__ == '__main__':
    sys.exit(main())