import easyocr
import re


class LicensePlateReader:
    """
    License plate reader for Vietnamese car license plates.
    
    Vietnamese License Plate Format:
    - Front plate (single line): NNL-NNN.NN (e.g., "75A-145.19")
      - NN: province/city code (2 digits, range 01-99)
      - L: series letter (A-Z, excluding I, O, Q)
      - NNN.NN: unique vehicle number (3 digits, dot, 2 digits)
    
    - Rear plate (two lines):
      - Line 1: NNL (province code + series letter)
      - Line 2: NNN.NN (unique vehicle number)
    
    Character set:
    - Digits: 0-9
    - Letters: A-Z (excluding I, O, Q)
    """
    
    # Valid letters for Vietnamese plates (excluding I, O, Q)
    VALID_LETTERS = set("ABCDEFGHJKLMNPRSTUVWXYZ")
    VALID_DIGITS = set("0123456789")
    
    # Province codes range from 01 to 99
    PROVINCE_CODE_MIN = 1
    PROVINCE_CODE_MAX = 99

    def __init__(self, use_gpu=False):
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        
        # Character correction mappings for OCR errors
        # When expecting a digit, these letters are commonly misread
        self.dict_char_to_int = {
            "O": "0", "Q": "0", "D": "0",  # Round shapes -> 0
            "I": "1", "L": "1",             # Vertical lines -> 1
            "Z": "2",                        # Z shape -> 2
            "J": "3",                        # J shape -> 3
            "A": "4",                        # A shape -> 4
            "S": "5",                        # S shape -> 5
            "G": "6", "C": "6",             # Round shapes -> 6
            "T": "7",                        # T shape -> 7
            "B": "8",                        # B shape -> 8
            "P": "9", "R": "9"              # P/R shape -> 9
        }
        
        # When expecting a letter, these digits are commonly misread
        self.dict_int_to_char = {
            "0": "D",  # 0 -> D (round shape)
            "1": "L",  # 1 -> L (vertical line)
            "2": "Z",  # 2 -> Z
            "3": "J",  # 3 -> J
            "4": "A",  # 4 -> A
            "5": "S",  # 5 -> S
            "6": "G",  # 6 -> G
            "7": "T",  # 7 -> T
            "8": "B",  # 8 -> B
            "9": "P"   # 9 -> P
        }
        
        # Excluded letters that might be misread
        self.excluded_letter_corrections = {
            "I": "L",  # I is excluded, replace with L
            "O": "D",  # O is excluded, replace with D
            "Q": "D"   # Q is excluded, replace with D
        }

    def _is_valid_digit(self, char):
        """Check if character is a valid digit or can be converted to one."""
        return char in self.VALID_DIGITS or char in self.dict_char_to_int
    
    def _is_valid_letter(self, char):
        """Check if character is a valid Vietnamese plate letter or can be converted to one."""
        return char in self.VALID_LETTERS or char in self.dict_int_to_char or char in self.excluded_letter_corrections
    
    def _convert_to_digit(self, char):
        """Convert character to digit if needed."""
        if char in self.VALID_DIGITS:
            return char
        return self.dict_char_to_int.get(char, char)
    
    def _convert_to_letter(self, char):
        """Convert character to valid Vietnamese plate letter if needed."""
        if char in self.VALID_LETTERS:
            return char
        if char in self.dict_int_to_char:
            return self.dict_int_to_char[char]
        if char in self.excluded_letter_corrections:
            return self.excluded_letter_corrections[char]
        return char

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with Vietnamese format.
        
        Supported formats:
        - Full format with separators: NNL-NNN.NN (10 chars) e.g., "75A-145.19"
        - Full format without separators: NNNNNNN (8 chars) e.g., "75A14519"
        - Two-line combined: NNNNNNN (8 chars) e.g., "75A14519"

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        # Remove common separators and spaces for validation
        cleaned = text.replace("-", "").replace(".", "").replace(" ", "")
        
        # Vietnamese plate should have 8 alphanumeric characters: NNL + NNNNN
        if len(cleaned) != 8:
            return False
        
        # Check format: NN (digits) + L (letter) + NNNNN (5 digits)
        # Position 0-1: Province code (2 digits)
        # Position 2: Series letter
        # Position 3-7: Vehicle number (5 digits)
        
        # Check positions 0-1 are digits (province code)
        if not (self._is_valid_digit(cleaned[0]) and self._is_valid_digit(cleaned[1])):
            return False
        
        # Check position 2 is a valid letter (series letter)
        if not self._is_valid_letter(cleaned[2]):
            return False
        
        # Check positions 3-7 are digits (vehicle number)
        for i in range(3, 8):
            if not self._is_valid_digit(cleaned[i]):
                return False
        
        # Validate province code range (01-99)
        province_code = int(self._convert_to_digit(cleaned[0]) + self._convert_to_digit(cleaned[1]))
        if not (self.PROVINCE_CODE_MIN <= province_code <= self.PROVINCE_CODE_MAX):
            return False
        
        return True

    def format_license(self, text):
        """
        Format the license plate text to Vietnamese standard format: NNL-NNN.NN
        
        Converts OCR-misread characters and formats with proper separators.
        - Positions 0-1: Province code (digits)
        - Position 2: Series letter
        - Positions 3-7: Vehicle number (digits)
        
        Output format: "NNL-NNN.NN" (e.g., "75A-145.19")

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text in Vietnamese format.
        """
        # Remove existing separators
        cleaned = text.replace("-", "").replace(".", "").replace(" ", "")
        
        if len(cleaned) != 8:
            return text  # Return original if unexpected length
        
        formatted = ""
        
        # Position 0-1: Province code (convert to digits)
        formatted += self._convert_to_digit(cleaned[0])
        formatted += self._convert_to_digit(cleaned[1])
        
        # Position 2: Series letter (convert to valid letter)
        formatted += self._convert_to_letter(cleaned[2])
        
        # Add dash separator
        formatted += "-"
        
        # Positions 3-5: First 3 digits of vehicle number
        formatted += self._convert_to_digit(cleaned[3])
        formatted += self._convert_to_digit(cleaned[4])
        formatted += self._convert_to_digit(cleaned[5])
        
        # Add dot separator
        formatted += "."
        
        # Positions 6-7: Last 2 digits of vehicle number
        formatted += self._convert_to_digit(cleaned[6])
        formatted += self._convert_to_digit(cleaned[7])
        
        return formatted
    
    def _try_parse_two_line_plate(self, detections):
        """
        Try to parse a two-line rear plate format.
        
        Rear plate format:
        - Line 1: NNL (province code + series letter)
        - Line 2: NNN.NN (vehicle number with dot)
        
        Args:
            detections: List of OCR detections.
            
        Returns:
            tuple: (combined_text, average_score) or (None, None)
        """
        if len(detections) < 2:
            return None, None
        
        # Sort detections by vertical position (top to bottom)
        sorted_detections = sorted(detections, key=lambda d: d[0][0][1])  # Sort by y coordinate
        
        for i in range(len(sorted_detections) - 1):
            line1_text = sorted_detections[i][1].upper().replace(" ", "")
            line1_score = sorted_detections[i][2]
            
            for j in range(i + 1, len(sorted_detections)):
                line2_text = sorted_detections[j][1].upper().replace(" ", "").replace(".", "")
                line2_score = sorted_detections[j][2]
                
                # Check if line1 matches NNL pattern (3 chars)
                if len(line1_text) == 3:
                    # Check if line2 matches NNNNN pattern (5 digits)
                    if len(line2_text) == 5:
                        combined = line1_text + line2_text
                        if self.license_complies_format(combined):
                            avg_score = (line1_score + line2_score) / 2
                            return combined, avg_score
        
        return None, None

    def _try_flexible_match(self, text):
        """
        Try to flexibly match Vietnamese license plate format using regex.
        This handles various OCR outputs with different separators and spacing.
        
        Args:
            text: Raw OCR text
            
        Returns:
            tuple: (cleaned_text, True) if matches pattern, (None, False) otherwise
        """
        # Clean the text: keep only alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Try to match 8-character Vietnamese format
        if len(cleaned) == 8:
            return cleaned, True
            
        # Try to extract pattern from longer strings (OCR might add noise)
        # Look for pattern: 2 digits + 1 letter + 5 digits
        pattern = r'(\d{2})([A-Z])(\d{5})'
        match = re.search(pattern, cleaned)
        if match:
            extracted = match.group(1) + match.group(2) + match.group(3)
            return extracted, True
        
        # Try alternative pattern with OCR errors (letter-like chars in digit positions)
        pattern_flexible = r'([0-9OQDI]{2})([A-Z0-9])([0-9OQDILSZJAGCTBPR]{5})'
        match = re.search(pattern_flexible, cleaned)
        if match:
            extracted = match.group(1) + match.group(2) + match.group(3)
            if len(extracted) == 8:
                return extracted, True
        
        return None, False

    def read(self, image):
        """
        Read the license plate text from the given cropped image.
        
        Supports Vietnamese license plate formats:
        - Front plate (single line): NNL-NNN.NN (e.g., "75A-145.19")
        - Rear plate (two lines): Line 1: NNL, Line 2: NNN.NN

        Args:
            image: Cropped image containing the license plate.

        Returns:
            tuple: Tuple containing the formatted license plate text and its confidence score.
                   Format: "NNL-NNN.NN" (e.g., "75A-145.19")
        """
        detections = self.reader.readtext(image)
        
        if not detections:
            return None, None

        # First, try to find a single-line detection that matches the format
        for detection in detections:
            bbox, text, score = detection
            text = text.upper().replace(" ", "")

            if self.license_complies_format(text):
                return self.format_license(text), score
        
        # Second, try flexible matching on each detection
        for detection in detections:
            bbox, text, score = detection
            flexible_text, matched = self._try_flexible_match(text)
            if matched and self.license_complies_format(flexible_text):
                return self.format_license(flexible_text), score
        
        # Third, try to combine all detected text and match
        all_text = "".join([d[1] for d in detections])
        avg_score = sum([d[2] for d in detections]) / len(detections)
        flexible_text, matched = self._try_flexible_match(all_text)
        if matched and self.license_complies_format(flexible_text):
            return self.format_license(flexible_text), avg_score
        
        # Fourth, try to combine detections for two-line plates
        combined_text, combined_score = self._try_parse_two_line_plate(detections)
        if combined_text:
            return self.format_license(combined_text), combined_score

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
    
    # Test format validation with Vietnamese plates
    print("\n--- Testing Vietnamese License Plate Format Validation ---")
    test_cases = [
        ("75A14519", True, "75A-145.19"),      # Valid: no separators
        ("75A-145.19", True, "75A-145.19"),    # Valid: with separators  
        ("30H12345", True, "30H-123.45"),      # Valid: Hanoi plate
        ("30E69937", True, "30E-699.37"),      # Valid: sample from user's image
        ("51G99999", True, "51G-999.99"),      # Valid: Ho Chi Minh plate
        ("75I14519", True, "75L-145.19"),      # Valid: I gets converted to L
        ("75O14519", True, "75D-145.19"),      # Valid: O gets converted to D
        ("00A12345", False, None),              # Invalid: province code 00
        ("ABC12345", False, None),              # Invalid: letters in province code
        ("75A1234", False, None),               # Invalid: too short (7 chars)
        ("75A123456", False, None),             # Invalid: too long (9 chars)
    ]
    
    for text, expected_valid, expected_format in test_cases:
        is_valid = license_reader.license_complies_format(text)
        formatted = license_reader.format_license(text) if is_valid else None
        status = "✓" if is_valid == expected_valid else "✗"
        print(f"  {status} '{text}' -> valid={is_valid}, formatted='{formatted}'")
        if is_valid and expected_format:
            format_status = "✓" if formatted == expected_format else "✗"
            print(f"    {format_status} Expected format: '{expected_format}'")
    
    # Test flexible matching
    print("\n--- Testing Flexible Matching ---")
    flexible_test_cases = [
        ("30E-699.37", "30E69937", True),       # With separators
        ("30E 699 37", "30E69937", True),       # With spaces
        ("30E.699.37", "30E69937", True),       # With dots
        ("ABC30E69937XYZ", "30E69937", True),   # With noise around
        ("30E699", None, False),                 # Too short
    ]
    
    for text, expected_cleaned, expected_match in flexible_test_cases:
        cleaned, matched = license_reader._try_flexible_match(text)
        status = "✓" if matched == expected_match and (cleaned == expected_cleaned or not expected_match) else "✗"
        print(f"  {status} '{text}' -> cleaned='{cleaned}', matched={matched}")

    # Load video
    video_path = os.path.join("data", "sample.mp4")
    print(f"\nOpening video file: {video_path}")
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
    print("Expected format: NNL-NNN.NN (e.g., '75A-145.19')")
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
            print(f"  License plate {i+1}: Could not read text (no Vietnamese format match)")

    # Clean up
    cap.release()
    print("\nTest completed.")


if __name__ == "__main__":
    # test()
    pass
