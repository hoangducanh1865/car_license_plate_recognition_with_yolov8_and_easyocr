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


def main(args):
    """
    Process video to detect vehicles and license plates.

    Args:
        max_frames (int, optional): Maximum number of frames to process.
                                    If None, processes all frames.
    """

    max_frames = args.max_frames
    use_kaggle = args.use_kaggle
    video_path = args.video_path
    output_video_path = args.output_video_path
    output_csv_path = args.output_csv_path
    interpolated_csv_path = args.interpolated_csv_path

    if use_kaggle:
        video_path = os.path.join("../../input/lpr-dataset", video_path)

    # Set device based on use_kaggle (GPU on Kaggle, CPU locally)
    device = "cuda" if use_kaggle else "cpu"
    use_gpu = bool(use_kaggle)
    print(f"Using device: {device}")

    # Initialize components
    print("Initializing components...")
    cap = cv2.VideoCapture(video_path)
    vehicle_detector = VehicleDetector(device=device)
    plate_detector = LicensePlateDetector(
        os.path.join("models", "license_plate_detector.pt"), device=device
    )
    tracker = VehicleTracker()
    reader = LicensePlateReader(use_gpu=use_gpu)
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

            # Preprocess the crop for better OCR
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Use adaptive thresholding for better results with varying lighting
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Try reading from processed image first, then fall back to original crop
            text, text_score = reader.read(processed)
            if text is None:
                # Fall back to original grayscale if adaptive threshold fails
                text, text_score = reader.read(gray)
            if text is None:
                # Final fallback: try original color crop
                text, text_score = reader.read(crop)
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
    ResultsWriter.write(results, output_csv_path)
    print(f"Results saved to: {output_csv_path}\n")

    # Interpolate missing data
    print("Interpolating missing data...")
    interpolator = BoundingBoxInterpolator()
    interpolator.interpolate_csv(output_csv_path, interpolated_csv_path)
    print(f"Interpolated results saved to: {interpolated_csv_path}\n")

    # Create visualization video
    print("Creating visualization video...")
    visualizer = VideoVisualizer()
    visualizer.visualize(interpolated_csv_path, video_path, output_video_path)
    print(f"Output video saved to: {output_video_path}\n")

    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car License Plate Recognition")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all frames)",
    )
    parser.add_argument(
        "--use_kaggle",
        type=int,
        default=0,
        help="Use Kaggle paths (1) or local paths (0) (default: 0)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=os.path.join("data", "test_videos", "test_2.mp4"),
        help="Path to test video",
    )
    parser.add_argument(
        "--output_video_path",
        type=str,
        default=os.path.join("results", "test_video_outputs", "test_i_output.mp4"),
        help="",
    )
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default=os.path.join("results", "test.csv"),
        help=")",
    )
    parser.add_argument(
        "--interpolated_csv_path",
        type=str,
        default=os.path.join("results", "test_interpolated.csv"),
        help="",
    )

    args = parser.parse_args()
    main(args=args)
