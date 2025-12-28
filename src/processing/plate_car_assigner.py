class PlateCarAssigner:
    """Assigns license plates to vehicles based on spatial containment."""

    @staticmethod
    def assign(plate, tracks):
        """
        Retrieve the vehicle coordinates and ID based on the license plate coordinates.

        Args:
            plate (tuple): Tuple containing the coordinates of the license plate
                          (x1, y1, x2, y2, score, class_id).
            tracks (list): List of vehicle track IDs and their corresponding coordinates.

        Returns:
            tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID,
                   or None if no matching vehicle is found.
        """
        x1, y1, x2, y2, *_ = plate

        for cx1, cy1, cx2, cy2, car_id in tracks:
            # Check if license plate is contained within vehicle bbox
            if x1 > cx1 and y1 > cy1 and x2 < cx2 and y2 < cy2:
                return cx1, cy1, cx2, cy2, car_id

        return None
