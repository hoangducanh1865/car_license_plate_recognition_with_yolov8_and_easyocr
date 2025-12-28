from ultralytics import YOLO


class LicensePlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)

    def detect(self, frame):

        # @QUESTION: Why [0]? How do we know to extract the license plate box list by this "boxes.data.tolist()"?
        # ANSWER: See in my notes
        return self.model(frame)[0].boxes.data.tolist()
