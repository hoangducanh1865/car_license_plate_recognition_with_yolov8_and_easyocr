import easyocr
import string


class LicensePlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
        self.dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.
        Format: LL-NN-LLL (L=Letter, N=Number)

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        if len(text) != 7:
            return False

        if (
            (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char.keys())
            and (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char.keys())
            and (
                text[2] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[2] in self.dict_char_to_int.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in self.dict_char_to_int.keys()
            )
            and (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys())
            and (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char.keys())
            and (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char.keys())
        ):
            return True
        else:
            return False

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.
        Positions 0,1,4,5,6 should be letters (convert numbers to letters)
        Positions 2,3 should be numbers (convert letters to numbers)

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ""
        mapping = {
            0: self.dict_int_to_char,
            1: self.dict_int_to_char,
            2: self.dict_char_to_int,
            3: self.dict_char_to_int,
            4: self.dict_int_to_char,
            5: self.dict_int_to_char,
            6: self.dict_int_to_char,
        }
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def read(self, image):
        """
        Read the license plate text from the given cropped image.

        Args:
            image: Cropped image containing the license plate.

        Returns:
            tuple: Tuple containing the formatted license plate text and its confidence score.
        """
        detections = self.reader.readtext(image)

        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(" ", "")

            if self.license_complies_format(text):
                return self.format_license(text), score

        return None, None


def test():
    import cv2
    import os
    from src.detectors.vehicle_detector import VehicleDetector
    from src.detectors.license_plate_detector import LicensePlateDetector

    # Initialize all components
    print("Initializing components...")
    license_reader = LicensePlateReader()
    vehicle_detector = VehicleDetector()
    license_plate_model_path = os.path.join("models", "license_plate_detector.pt")
    license_plate_detector = LicensePlateDetector(str(license_plate_model_path))
    print("All components initialized.")

    # Load video
    video_path = os.path.join("data", "sample.mp4")
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    # Read the first frame
    print("Reading the first frame...")
    ret, frame = cap.read()

    # Step 1: Detect vehicles
    print("\nStep 1: Detecting vehicles...")
    vehicle_detections = vehicle_detector.detect(frame)
    print(f"Found {len(vehicle_detections)} vehicles")

    # Step 2: Detect license plates in the frame
    print("\nStep 2: Detecting license plates...")
    license_plate_detections = license_plate_detector.detect(frame)
    print(f"Found {len(license_plate_detections)} license plates")

    # Step 3: Read license plates
    print("\nStep 3: Reading license plate text...")
    for i, lp_detection in enumerate(license_plate_detections):
        x1, y1, x2, y2, score = lp_detection[:5]

        # Crop the license plate region from the frame
        lp_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]

        if lp_crop.size == 0:
            print(f"  License plate {i+1}: Empty crop, skipping")
            continue

        # Read the license plate text
        text, conf = license_reader.read(lp_crop)

        if text:
            print(f"  License plate {i+1}:")
            print(f"    Bounding box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
            print(f"    Detection score: {score:.2f}")
            print(f"    Text: {text}")
            print(f"    OCR confidence: {conf:.2f}")
        else:
            print(f"  License plate {i+1}: Could not read text (no 7-character match)")

    # Clean up
    cap.release()
    print("\nTest completed.")


if __name__ == "__main__":
    # test()
    pass
