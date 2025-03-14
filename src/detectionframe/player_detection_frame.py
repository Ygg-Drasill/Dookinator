import numpy as np

class PlayerDetectionFrame:
    def __init__(self, player_id: str, number: int, xyz: np.array, speed: float, opta_id: int):
        self.player_id = player_id
        self.number = number
        self.xyz = xyz  # Expected to be a NumPy array with (x, y, z) coordinates
        self.speed = speed
        self.opta_id = opta_id

    def to_dict(self):
        return {
            "player_id": self.player_id,
            "number": self.number,
            "xyz": self.xyz.tolist(),  # Convert NumPy array to list for serialization
            "speed": self.speed,
            "opta_id": self.opta_id
        }