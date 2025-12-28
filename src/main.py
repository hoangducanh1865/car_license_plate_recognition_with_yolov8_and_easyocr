import os
import cv2
import argparse

from src.detectors.vehicle_detector import VehicleDetector
from src.detectors.license_plate_detector import LicensePlateDetector
from src.tracking.vehicle_tracker import VehicleTracker
from src.ocr.license_plate_reader import LicensePlateReader
from src.processing.plate_car_assigner import PlateCarAssigner
from src.processing.results_writer import ResultsWriter
from src.processing.interpolator import BoundingBoxInterpolator
from src.visualization.video_visualizer import VideoVisualizer


def main(max_frames=None, use_kaggle=0):
    """
    Process video to detect vehicles and license plates.

    Args:
        max_frames (int, optional): Maximum number of frames to process.
                                    If None, processes all frames.
    """
    # Paths
    video_path = os.path.join("..", "..", "input", "sample", "sample.mp4") if use_kaggle else os.path.join("data", "sample.mp4")
    output_csv = os.path.join("results", "test.csv")
    interpolated_csv = os.path.join("results", "test_interpolated.csv")
    output_video = os.path.join("results", "out.mp4") 

    # Initialize components
    print("Initializing components...")
    cap = cv2.VideoCapture(video_path)
    vehicle_detector = VehicleDetector()
    plate_detector = LicensePlateDetector(os.path.join("models", "license_plate_detector.pt"))
    tracker = VehicleTracker()
    reader = LicensePlateReader()
    print("Components initialized.\n")

    # Results dictionary with nested structure
    results = {}
    frame_id = 0

    # Determine frame limit message
    if max_frames is not None:
        print(f"Processing up to {max_frames} video frames...")
    else:
        print("Processing all video frames...")

    # Process video frames
    while True:
        # Check if we've reached the frame limit
        if max_frames is not None and frame_id >= max_frames:
            print(f"Reached frame limit of {max_frames} frames.")
            break

        ret, frame = cap.read()
        if not ret:
            break

        results[frame_id] = {}

        # Detect vehicles and track them
        vehicles = vehicle_detector.detect(frame)
        tracked = tracker.update(vehicles)

        # Detect license plates
        plates = plate_detector.detect(frame)

        for plate in plates:
            # Assign plate to car
            assignment = PlateCarAssigner.assign(plate, tracked)
            if assignment is None:
                continue

            xcar1, ycar1, xcar2, ycar2, car_id = assignment
            x1, y1, x2, y2, score, _ = plate

            # Crop license plate
            crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                continue

            # Optional: preprocess the crop
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate text
            text, text_score = reader.read(thresh)
            if text is None:
                continue

            # Store results in the expected nested structure
            results[frame_id][car_id] = {
                "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                "license_plate": {
                    "bbox": [x1, y1, x2, y2],
                    "bbox_score": score,
                    "text": text,
                    "text_score": text_score,
                },
            }

        frame_id += 1

        # Print progress
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")

    cap.release()
    print(f"Processed {frame_id} frames total.\n")

    # Write results to CSV
    print("Writing results to CSV...")
    ResultsWriter.write(results, output_csv)
    print(f"Results saved to: {output_csv}\n")

    # Interpolate missing data
    print("Interpolating missing data...")
    interpolator = BoundingBoxInterpolator()
    interpolator.interpolate_csv(output_csv, interpolated_csv)
    print(f"Interpolated results saved to: {interpolated_csv}\n")

    # Create visualization video
    print("Creating visualization video...")
    visualizer = VideoVisualizer()
    visualizer.visualize(interpolated_csv, video_path, output_video)
    print(f"Output video saved to: {output_video}\n")

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car License Plate Recognition")
    parser.add_argument("--max_frames", type=int, default=None, 
                        help="Maximum number of frames to process (default: all frames)")
    parser.add_argument("--use_kaggle", type=int, default=0, 
                        help="Use Kaggle paths (1) or local paths (0) (default: 0)")
    
    args = parser.parse_args()
    main(max_frames=args.max_frames, use_kaggle=args.use_kaggle)
