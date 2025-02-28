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
    results = model(frame)[0]  # Return the first result (not a list)



    return results  # Keep the original result object
