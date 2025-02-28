from supervision import Detections
import numpy as np

selected_classes = [0, 1, 2]
min_confidence = 0.5

def filter_detections(detections: Detections) -> Detections:
    """Filter detections based on detection confidence and class id.

        Args:
            detections (Detections): detections to filter.

        Returns:
            Detections: Filtered detections.
        """
    detections = detections[np.isin(detections.class_id, selected_classes)]

    detections = detections[detections.confidence >= min_confidence]

    return detections
