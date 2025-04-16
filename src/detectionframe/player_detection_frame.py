import numpy as np

class PlayerDetectionFrame:
    def __init__(self, player_id: str, number: int, xyz: np.array, speed: float, opta_id: int):
        self.player_id = player_id
        self.number = number
        self.xyz = xyz
        self.speed = speed
        self.opta_id = opta_id

    def to_dict(self):
        return {
            "playerId": self.player_id,
            "number": self.number,
            "xyz": self.xyz.tolist(),
            "speed": self.speed,
            "optaId": self.opta_id
        }