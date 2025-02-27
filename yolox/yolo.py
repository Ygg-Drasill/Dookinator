from ultralytics import YOLO
import torch
import numpy as np

CONFIDENCE_THRESHOLD = 0.4
REFEREE_CLASS_ID = 3


def init_yolo(model_path: str = "yolov8n-football.pt") -> YOLO:
    """Initialize the YOLO model with GPU support if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return YOLO(model_path).to(device)


def read_frame(model: YOLO, frame: np.ndarray) -> list:
    """Process a frame using the YOLO model and filter detections.

    Args:
        model (YOLO): The YOLO model instance.
        frame (np.ndarray): The input image/frame.

    Returns:
        list: Filtered detections in the format [[x, y, w, h], confidence, class_id].
    """
    detections = model(frame)[0].boxes.data.tolist()

    results = []
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection

        # Filter out weak detections with low confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # Get the bounding box and the class id
        bbox = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
        class_id = int(class_id)

        # Apply confidence threshold and filter out referees
        if class_id != REFEREE_CLASS_ID:
            results.append([bbox, confidence, class_id])

    return results
