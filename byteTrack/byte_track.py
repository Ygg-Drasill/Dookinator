import yaml
from supervision import Detections
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    yolo = config['yolo']

selected_classes = [0, 1, 2]
min_confidence = 0.5

def filter_detections(detections: Detections) -> Detections:
    """Filter detections based on detection confidence and class id.

        Args:
            detections (Detections): detections to filter.

        Returns:
            Detections: Filtered detections.
        """
    detections = detections[np.isin(detections.class_id, list(yolo['selected_class_ids'].values()))]

    detections = detections[detections.confidence >= yolo['confidence_threshold']]

    return detections
