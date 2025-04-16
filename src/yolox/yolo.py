import os

from supervision import KeyPoints
from ultralytics import YOLO
import torch
import numpy as np
from ultralytics.engine.results import Results
import supervision as sv

class SoccerYOLOX:

    def __init__(self, soccer_model_path: str, keypoint_model_path: str):
        """Initialize the YOLO models with GPU support if available."""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.soccer_model = YOLO(soccer_model_path).to(device)
        self.keypoint_model = YOLO(keypoint_model_path).to(device)

    def read_frame(self, frame: np.ndarray) -> Results:
        """Process a frame using the YOLO soccer model.

        Args:
            model (YOLO): The YOLO model instance.
            frame (np.ndarray): The input image/frame.

        Returns:
            Results: Detections of the frame.
        """
        results = self.soccer_model(frame)[0]

        return results

    def find_keypoint(self, frame: np.ndarray) -> KeyPoints:

        result = self.keypoint_model(frame)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        return keypoints