import json
import os
import sys

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
    pitch_data = json.load(f)

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

    #while True:

    read_next_frames()

    #if cv2.waitKey(1) == ord("q"):
        #break

    pass
    video_cap.release()
    #cv2.destroyAllWindows()
    return 0

def read_next_frames():
    global frame_count

    number_of_frames_to_read = 4500  # Number of frames to read
    frames = []
    detections_chunk = []
    key_points = []

    for i in range(number_of_frames_to_read):
        ret, frame = video_cap.read()
        if not ret:
            print("Could not capture video frame.")
            break

        frames.append(frame)
        results = soccer_YOLOX.read_frame(frame)
        key_points.append(soccer_YOLOX.find_keypoint(frame))

        detections_chunk.append(sv.Detections.from_ultralytics(results))

        detections_chunk[i] = filter_detections(detections_chunk[i])

        detections_chunk[i] = tracker.update_with_detections(detections_chunk[i])

        pass

    detections_chunk = player_separation(frames, detections_chunk)

    ######################################
    # Output
    ######################################

    for i in range(number_of_frames_to_read):

        detection_frame = calculate_detection_frame(detections_chunk[i], frame_count, key_points[i])

        output_detection_frame(detection_frame, os.path.join(ROOT_DIR, str(jsonl_file_path)))

        # Create labels with tracker IDs
        labels = [
            f"#{tracker_id} + {team}"
            for j, (tracker_id, team) in enumerate(zip(detections_chunk[i].tracker_id, detections_chunk[i]["player_team"]))
        ]

        annotated_frame = box_annotator.annotate(
            frames[i].copy(), detections=detections_chunk[i])

        label_annotator.annotate(
            annotated_frame, detections=detections_chunk[i], labels=labels)

        pitch_length = pitch_data["pitchLength"]
        pitch_width = pitch_data["pitchWidth"]

        if config['show_output_overlay']:
            annotated_frame = draw_overlay(SoccerPitchConfiguration(width=pitch_width, length=pitch_length),
                                           annotated_frame, detection_frame, scale=3)

        soccer_video_writer.write_frame(annotated_frame)
        # show the frame to our screen
        #cv2.imshow("Frame", annotated_frame)

        frame_count += 1

    pass

if __name__ == '__main__':
    sys.exit(main())