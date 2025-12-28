import ast
import cv2
import numpy as np
import pandas as pd


class VideoVisualizer:
    """Visualizes detection results on video with license plates."""

    def __init__(self):
        self.license_plate_cache = {}

    @staticmethod
    def draw_border(
        img,
        top_left,
        bottom_right,
        color=(0, 255, 0),
        thickness=10,
        line_length_x=200,
        line_length_y=200,
    ):
        """
        Draw stylized corner borders around a bounding box.

        Args:
            img: Image to draw on
            top_left: Top-left corner coordinates (x1, y1)
            bottom_right: Bottom-right corner coordinates (x2, y2)
            color: Border color (B, G, R)
            thickness: Line thickness
            line_length_x: Length of horizontal border lines
            line_length_y: Length of vertical border lines

        Returns:
            Modified image with borders drawn
        """
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Top-left corner
        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

        return img

    def _extract_best_license_plates(self, results, cap):
        """
        Extract the best license plate crop for each car based on highest confidence score.
        Filters out interpolated frames (where license_number_score is '0').

        Args:
            results: DataFrame with detection results
            cap: Video capture object

        Returns:
            Dictionary mapping car_id to license plate info
        """
        license_plate = {}

        for car_id in np.unique(results["car_id"]):
            # Filter car data and convert score to numeric, excluding interpolated frames
            car_data = results[results["car_id"] == car_id].copy()
            car_data["license_number_score"] = pd.to_numeric(
                car_data["license_number_score"], errors="coerce"
            )

            # Filter out interpolated frames (score = 0) and invalid license numbers
            valid_data = car_data[
                (car_data["license_number_score"] > 0)
                & (car_data["license_number"] != "0")
                & (car_data["license_number"].notna())
            ]

            if len(valid_data) == 0:
                print(f"Warning: No valid license plate data found for car_id {car_id}")
                continue

            # Get the row with the highest confidence score
            best_row = valid_data.loc[valid_data["license_number_score"].idxmax()]
            max_score = best_row["license_number_score"]
            license_number = best_row["license_number"]
            frame_nmr = int(best_row["frame_nmr"])
            bbox_str = best_row["license_plate_bbox"]

            print(
                f"Car {car_id}: Best license plate = '{license_number}' (score: {max_score:.3f}, frame: {frame_nmr})"
            )

            license_plate[car_id] = {"license_crop": None, "license_plate_number": license_number}

            # Get frame with best detection
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
            ret, frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_nmr} for car_id {car_id}")
                continue

            # Extract license plate bbox
            x1, y1, x2, y2 = ast.literal_eval(
                bbox_str.replace("[ ", "[").replace("   ", " ").replace("  ", " ").replace(" ", ",")
            )

            # Crop and resize license plate
            license_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

            license_plate[car_id]["license_crop"] = license_crop

        return license_plate

    def _parse_bbox(self, bbox_str):
        """
        Parse bounding box string to coordinates.

        Args:
            bbox_str: String representation of bbox like '[x1 y1 x2 y2]'

        Returns:
            Tuple of (x1, y1, x2, y2)
        """
        return ast.literal_eval(
            bbox_str.replace("[ ", "[").replace("   ", " ").replace("  ", " ").replace(" ", ",")
        )

    def _overlay_license_plate_text(self, frame, license_bbox, license_number):
        """
        Overlay license plate text box above the license plate with connecting lines.
        Only shows text, no car/license plate image.

        Args:
            frame: Frame to draw on
            license_bbox: License plate bounding box (x1, y1, x2, y2)
            license_number: License plate text
        """
        lp_x1, lp_y1, lp_x2, lp_y2 = license_bbox
        frame_height, frame_width = frame.shape[:2]

        # Calculate center of license plate
        lp_center_x = int((lp_x1 + lp_x2) / 2)

        try:
            # Calculate text size to determine text box dimensions
            font_scale = 2.0
            font_thickness = 6
            (text_width, text_height), baseline = cv2.getTextSize(
                license_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Add padding around text
            text_padding = 20
            text_box_width = text_width + 2 * text_padding
            text_box_height = text_height + baseline + 2 * text_padding

            # Calculate text box position (above the license plate)
            text_box_y2 = max(int(lp_y1) - 30, text_box_height + 30)
            text_box_y1 = text_box_y2 - text_box_height

            # Center the text box horizontally with the license plate
            text_box_x1 = lp_center_x - text_box_width // 2
            text_box_x2 = text_box_x1 + text_box_width

            # Ensure text box stays within frame bounds
            if text_box_y1 < 0:
                text_box_y1 = 10
                text_box_y2 = text_box_y1 + text_box_height
            if text_box_x1 < 0:
                text_box_x1 = 10
                text_box_x2 = text_box_x1 + text_box_width
            elif text_box_x2 > frame_width:
                text_box_x2 = frame_width - 10
                text_box_x1 = text_box_x2 - text_box_width

            # Draw white background for text box
            cv2.rectangle(
                frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (255, 255, 255), -1
            )  # -1 fills the rectangle

            # Draw border around text box
            cv2.rectangle(
                frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (0, 0, 0), 2
            )

            # Draw license plate text centered in text box
            text_x = text_box_x1 + text_padding
            text_y = text_box_y1 + text_padding + text_height
            cv2.putText(
                frame,
                license_number,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

            # Draw connecting lines from license plate to text box
            lp_top_center = (lp_center_x, int(lp_y1))
            text_box_bottom_center = (lp_center_x, text_box_y2)

            # Draw cyan line
            cv2.line(frame, lp_top_center, text_box_bottom_center, (255, 255, 0), 2)

        except Exception as e:
            print(f"Error overlaying license plate text: {e}")

    def _overlay_license_plate(self, frame, license_bbox, license_crop, license_number):
        """
        Overlay enlarged license plate crop with text above the detection.
        Auto-sizes the text box to fit the license plate text.

        Args:
            frame: Frame to draw on
            license_bbox: License plate bounding box (x1, y1, x2, y2)
            license_crop: Cropped license plate image
            license_number: License plate text
        """
        lp_x1, lp_y1, lp_x2, lp_y2 = license_bbox
        H, W, _ = license_crop.shape
        frame_height, frame_width = frame.shape[:2]

        # Calculate center of license plate
        lp_center_x = int((lp_x1 + lp_x2) / 2)

        try:
            # Calculate text size to determine text box dimensions
            font_scale = 3.0
            font_thickness = 10
            (text_width, text_height), baseline = cv2.getTextSize(
                license_number, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Add padding around text
            text_padding = 40
            text_box_width = text_width + 2 * text_padding
            text_box_height = text_height + baseline + 2 * text_padding

            # Resize license crop to match text box width
            aspect_ratio = W / H
            viz_width = text_box_width
            viz_height = int(viz_width / aspect_ratio)
            license_crop_resized = cv2.resize(license_crop, (viz_width, viz_height))

            # Total visualization height (text box + license crop)
            total_viz_height = viz_height + text_box_height

            # Calculate visualization position (above the license plate)
            viz_y2 = max(int(lp_y1) - 50, total_viz_height + 50)
            viz_y1 = viz_y2 - total_viz_height

            # Center the visualization horizontally with the license plate
            viz_x1 = lp_center_x - viz_width // 2
            viz_x2 = viz_x1 + viz_width

            # Ensure visualization stays within frame bounds
            if viz_y1 < 0:
                viz_y1 = 50
                viz_y2 = viz_y1 + total_viz_height
            if viz_x1 < 0:
                viz_x1 = 10
                viz_x2 = viz_x1 + viz_width
            elif viz_x2 > frame_width:
                viz_x2 = frame_width - 10
                viz_x1 = viz_x2 - viz_width

            # Calculate positions for text box and license crop
            text_box_y1 = viz_y1
            text_box_y2 = viz_y1 + text_box_height
            lp_crop_y1 = text_box_y2
            lp_crop_y2 = viz_y2

            # Draw white background for text box
            frame[text_box_y1:text_box_y2, viz_x1:viz_x2, :] = (255, 255, 255)

            # Place resized license plate crop below text
            frame[lp_crop_y1:lp_crop_y2, viz_x1:viz_x2, :] = license_crop_resized

            # Draw license plate text centered in text box
            text_x = viz_x1 + text_padding
            text_y = text_box_y1 + text_padding + text_height
            cv2.putText(
                frame,
                license_number,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

            # Draw connecting lines from license plate to visualization
            lp_top_left = (int(lp_x1), int(lp_y1))
            lp_top_right = (int(lp_x2), int(lp_y1))
            viz_bottom_left = (viz_x1, viz_y2)
            viz_bottom_right = (viz_x2, viz_y2)

            # Draw cyan lines
            cv2.line(frame, lp_top_left, viz_bottom_left, (255, 255, 0), 3)
            cv2.line(frame, lp_top_right, viz_bottom_right, (255, 255, 0), 3)

        except Exception as e:
            print(f"Error overlaying license plate: {e}")

    def visualize(self, csv_path, video_path, output_path):
        """
        Create visualization video with detected license plates.

        Args:
            csv_path: Path to interpolated CSV results
            video_path: Path to input video
            output_path: Path to output video
        """
        results = pd.read_csv(csv_path)

        # Get the maximum frame number in results
        max_frame_in_results = results["frame_nmr"].max()
        print(f"CSV contains data for frames 0-{max_frame_in_results}")

        # Load video
        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get best license plate crop for each car
        self.license_plate_cache = self._extract_best_license_plates(results, cap)

        # Reset to beginning
        frame_nmr = -1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Process frames - only up to the max frame in results
        ret = True
        while ret and frame_nmr < max_frame_in_results:
            ret, frame = cap.read()
            frame_nmr += 1

            if ret:
                df_frame = results[results["frame_nmr"] == frame_nmr]

                # Only process if we have data for this frame
                if len(df_frame) > 0:
                    for row_indx in range(len(df_frame)):
                        # Get license plate bounding box
                        x1, y1, x2, y2 = self._parse_bbox(
                            df_frame.iloc[row_indx]["license_plate_bbox"]
                        )

                        # Draw small rectangle around license plate only (not the car)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

                        # Overlay license plate text only
                        car_id = df_frame.iloc[row_indx]["car_id"]
                        if car_id in self.license_plate_cache:
                            license_number = self.license_plate_cache[car_id][
                                "license_plate_number"
                            ]

                            self._overlay_license_plate_text(
                                frame, (x1, y1, x2, y2), license_number
                            )

                out.write(frame)
                frame = cv2.resize(frame, (1280, 720))

        out.release()
        cap.release()

        print(
            f"Visualization complete! Processed {frame_nmr + 1} frames. Video saved to: {output_path}"
        )
