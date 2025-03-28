import os

from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.engine.results import Results

from src.definitions import ROOT_DIR


def init_yolo(model_path: str) -> YOLO:
    """Initialize the YOLO model with GPU support if available."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return YOLO(model_path).to(device)


def read_frame(model: YOLO, frame: np.ndarray) -> Results:
    """Process a frame using the YOLO model.

    Args:
        model (YOLO): The YOLO model instance.
        frame (np.ndarray): The input image/frame.

    Returns:
        Results: Detections of the frame.
    """
    results = model(frame)[0]  # Return the first result

    return results
