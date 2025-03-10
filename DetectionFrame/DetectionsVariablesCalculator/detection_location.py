import numpy as np


def get_screen_location_of_bounding_box(xyxy: np.ndarray) -> np.ndarray:
    """
    Computes the screen location of a bounding box given its (x1, y1, x2, y2) coordinates.

    Args:
        xyxy (np.ndarray): A 1D NumPy array of shape (4,) representing (x1, y1, x2, y2).

    Returns:
        np.ndarray: A 1D NumPy array with the midpoint x-coordinate and bottom y-coordinate.
    """
    mid_x = np.mean(xyxy[[0, 2]])  # Average of x1 and x2
    bottom_y = xyxy[3]

    return np.array([mid_x, bottom_y])
