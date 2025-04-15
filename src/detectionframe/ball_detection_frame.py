import numpy as np

class BallDetectionFrame:
    def __init__(self, xyz: np.ndarray, speed: float):
        self.xyz = np.array(xyz)
        self.speed = speed
        pass

    def to_dict(self):
        """Converts the object attributes to a dictionary."""
        return {
            "xyz": self.xyz.tolist(),
            "speed": self.speed,
        }
