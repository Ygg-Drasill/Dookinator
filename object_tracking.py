import datetime
import cv2
import supervision as sv

from byteTrack.byte_track import filter_detections
from yolox.yolo import init_yolo, read_frame

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# initialize the video capture object
video_cap = cv2.VideoCapture("full.mp4")

tracker = sv.ByteTrack(track_activation_threshold=0.5, lost_track_buffer=25)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

model = init_yolo()

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

    # Create labels with class names and tracker IDs
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