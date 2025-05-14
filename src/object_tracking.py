import json
import os
import sys

import numpy as np
import yaml

import cv2
import supervision as sv

from src.bytetrack.byte_track import filter_detections
from src.definitions import ROOT_DIR, CONFIG_PATH
from src.detectionframe.calculate_frame import calculate_detection_frame
from src.detectionframe.output_to_jsonl import output_detection_frame
from src.fieldPitch.SoccerPitchConfiguration import SoccerPitchConfiguration
from src.output.common import draw_overlay
from src.output.output_video_file import SoccerVideoWriter
from src.player_seperation import player_separation
from yolox.yolo import SoccerYOLOX

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    match = config['match']
    byte_track = config['byte_track']
    yolo = config['yolo']
    jsonl_file_path = config['output_jsonl_relative_file_path']

with open(os.path.join(ROOT_DIR, str(match['meta_file'])), 'r') as f:
    meta_data = json.load(f)

# initialize the video capture object
video_cap = cv2.VideoCapture(os.path.join(ROOT_DIR, str(match["video"])))

tracker = sv.ByteTrack(byte_track['track_activation_threshold'], byte_track['lost_track_buffer'],
                       byte_track['minimum_matching_threshold'], byte_track['frame_rate'],
                       byte_track['minimum_consecutive_frames'])
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

soccer_YOLOX = SoccerYOLOX(os.path.join(ROOT_DIR, str(yolo['soccer_model'])),
                           os.path.join(ROOT_DIR, str(yolo['keypoint_model'])))

# remove old output file
if os.path.exists(os.path.join(ROOT_DIR, str(jsonl_file_path))):
    os.remove(os.path.join(ROOT_DIR, str(jsonl_file_path)))

frame_count = 0

soccer_video_writer = SoccerVideoWriter()

def main() -> int:

    read_next_frames()

    video_cap.release()
    return 0

def read_next_frames():
    global frame_count

    number_of_frames_to_read = int(match['number_of_seconds_to_read'] * meta_data["fps"])
    frames = []
    detections_chunk = []
    key_point_chunk = []

    for i in range(number_of_frames_to_read):
        ret, frame = video_cap.read()
        if not ret:
            print("Could not capture video frame.")
            break

        frames.append(frame)
        results = soccer_YOLOX.read_frame(frame)

        key_points = soccer_YOLOX.find_keypoint(frame)

        key_point_chunk.append(key_points)

        detections_chunk.append(sv.Detections.from_ultralytics(results))

        ball_mask = detections_chunk[i].class_id == 0
        ball_detections = detections_chunk[i][ball_mask]
        ball_detections.tracker_id = np.full(len(ball_detections), -1, dtype=int)

        detections_chunk[i] = filter_detections(detections_chunk[i])

        detections_chunk[i] = tracker.update_with_detections(detections_chunk[i])

        ball_present_mask = detections_chunk[i].class_id == 0

        if len(ball_detections) > 0 and not np.any(ball_present_mask):
            detections_chunk[i] = sv.Detections.merge([detections_chunk[i], ball_detections])

        pass

    detections_chunk = player_separation(frames, detections_chunk)

    ######################################
    # Output
    ######################################

    for i, detections in enumerate(detections_chunk):

        try:
            detection_frame = calculate_detection_frame(detections, frame_count, key_point_chunk[i])
        except ValueError as error:
            print('Error: ' + repr(error))
            frame_count += 1
            continue

        output_detection_frame(detection_frame, os.path.join(ROOT_DIR, str(jsonl_file_path)))

        # Create labels with tracker IDs
        labels = [
            f"#{tracker_id} + {team}"
            for j, (tracker_id, team) in enumerate(zip(detections.tracker_id, detections["player_team"]))
        ]

        annotated_frame = box_annotator.annotate(
            frames[i].copy(), detections=detections)

        label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels)

        pitch_length = meta_data["pitchLength"]
        pitch_width = meta_data["pitchWidth"]

        if config['show_output_overlay']:
            annotated_frame = draw_overlay(SoccerPitchConfiguration(width=pitch_width, length=pitch_length),
                                           annotated_frame, detection_frame, scale=3)

        soccer_video_writer.write_frame(annotated_frame)

        frame_count += 1

    pass

if __name__ == '__main__':
    sys.exit(main())