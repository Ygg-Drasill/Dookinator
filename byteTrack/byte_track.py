import yaml
from supervision import Detections
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)  # Read the file once
    yolo = config['yolo']

def filter_detections(detections: Detections) -> Detections:
    """Filter detections based on detection confidence and class id.

        Args:
            detections (Detections): detections to filter.

        Returns:
            Detections: Filtered detections.
        """
    class_ids = [item["id"] for item in yolo["selected_class_ids"].values()]

    detections = detections[np.isin(detections.class_id, class_ids)]

    detections = detections[
        ((detections.confidence >= yolo["selected_class_ids"]['ball']['confidence_threshold']) | (
                    detections.class_id != 0)) &
        ((detections.confidence >= yolo["selected_class_ids"]['goalkeeper']['confidence_threshold']) | (
                    detections.class_id != 1)) &
        ((detections.confidence >= yolo["selected_class_ids"]['player']['confidence_threshold']) | (
                    detections.class_id != 2))
        ]

    return detections
