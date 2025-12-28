import os
from ultralytics import YOLO
from pathlib import Path


class VehicleDetector:
    def __init__(self, model_path=os.path.join("models", "license_plate_detector.pt")):
        self.model = YOLO(model=model_path)
        self.vehicale_classes = {2, 3, 5, 7}

    def detect(self, frame):
        detections_ = []
        detections = self.model(frame)[0]

        # @QUESTION: Well how do we known to extract the data list like this "detections.boxes.data.tolist()" :))?
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist():
            if class_id in self.vehicale_classes:
                detections_.append([x1, y1, x2, y2, score])

        return detections_


def test():
    import cv2
    import os
    from pathlib import Path

    # Initialize the VehicleDetector
    print("Initializing VehicleDetector...")
    vd = VehicleDetector()
    print("VehicleDetector initialized.")

    # Get the project root directory (two levels up from this file)
    current_dir = Path(__file__).resolve().parent  # src/detectors/
    project_root = current_dir.parent.parent  # project root
    video_path = project_root / "data" / "sample.mp4"

    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        print(f"File exists: {video_path.exists()}")
        return

    # Read the first frame
    print("Reading the first frame...")
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame of the video")
        return

    # Detect vehicles in the first frame
    print("Detecting vehicles...")
    detections = vd.detect(frame)

    # Print the detections
    print(f"Detections: {len(detections)} vehicles found")
    for i, det in enumerate(detections):
        print(f"  Vehicle {i+1}: {det}")

    # Release the video capture object
    cap.release()


if __name__ == "__main__":
    # test()
    pass
