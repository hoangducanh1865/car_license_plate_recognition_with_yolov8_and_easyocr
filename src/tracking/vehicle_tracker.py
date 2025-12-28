import numpy as np
from src.sort.sort import Sort


class VehicleTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        if len(detections) == 0:
            return []
        return self.tracker.update(np.array(detections))
