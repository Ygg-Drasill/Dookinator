import datetime
import yaml

import cv2
import supervision as sv


from byteTrack.byte_track import filter_detections
from player_separation import player_separation
from yolox.yolo import init_yolo, read_frame


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    video = config['video']
    byte_track = config['byte_track']
    yolo = config['yolo']

# initialize the video capture object
video_cap = cv2.VideoCapture(video)


tracker = sv.ByteTrack(byte_track['track_activation_threshold'], byte_track['lost_track_buffer'], byte_track['minimum_matching_threshold'], byte_track['frame_rate'], byte_track['minimum_consecutive_frames'])
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

model = init_yolo(yolo['model'])

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    results = read_frame(model, frame)

    ######################################
    # TRACKING
    ######################################

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    detections = filter_detections(detections)
    player_colors = player_separation(frame, detections.xyxy)

    # Create labels with tracker IDs
    labels = [
        f"#{tracker_id[0]}"
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

video_cap.release()
cv2.destroyAllWindows()