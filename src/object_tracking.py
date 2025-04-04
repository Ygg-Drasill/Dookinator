import datetime
import os

import yaml

import cv2
import supervision as sv

from src.bytetrack.byte_track import filter_detections
from src.definitions import ROOT_DIR, CONFIG_PATH
from src.detectionframe.calculate_frame import calculate_detection_frame
from src.detectionframe.output_to_jsonl import output_detection_frame
from yolox.yolo import init_yolo, read_frame


with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    video_path = config['video']
    byte_track = config['byte_track']
    yolo = config['yolo']
    jsonl_file_path = config['output_jsonl_relative_file_path']

# initialize the video capture object
video_cap = cv2.VideoCapture(os.path.join(ROOT_DIR, str(video_path)))

tracker = sv.ByteTrack(byte_track['track_activation_threshold'], byte_track['lost_track_buffer'], byte_track['minimum_matching_threshold'], byte_track['frame_rate'], byte_track['minimum_consecutive_frames'])
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

model = init_yolo(os.path.join(ROOT_DIR, str(yolo['model'])))

frame_count = 0

#remove old output file
if os.path.exists(os.path.join(ROOT_DIR, str(jsonl_file_path))):
    os.remove(os.path.join(ROOT_DIR, str(jsonl_file_path)))

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        print("Could not capture video frame.")
        break

    results = read_frame(model, frame)

    ######################################
    # TRACKING
    ######################################

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    detections = filter_detections(detections)
    player_colors = player_separation(frame, detections.xyxy, detections.tracker_id)

    detections = tracker.update_with_detections(detections)

    detection_frame = calculate_detection_frame(detections, frame_count)

    output_detection_frame(detection_frame, os.path.join(ROOT_DIR, str(jsonl_file_path)))

    # Create labels with tracker IDs
    labels = [
        f"#{tracker_id[0]}+{player_colors[tracker_id[0]]}"
        for tracker_id
        in zip(detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)

    label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    # show the frame to our screen
    cv2.imshow("Frame", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

    frame_count += 1

    pass
video_cap.release()
cv2.destroyAllWindows()