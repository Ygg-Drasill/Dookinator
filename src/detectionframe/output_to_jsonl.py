from src.detectionframe.game_detection_frame import GameDetectionFrame
import json

def output_detection_frame(game_detection_frame: GameDetectionFrame, file_path: str):
    with open(file_path, "a") as file:
        json.dump(game_detection_frame.to_dict(), file)
        file.write("\n")
    pass