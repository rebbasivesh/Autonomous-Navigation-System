from ultralytics import YOLO

class Detector:
    """
    A wrapper for the YOLO object detection model.
    """
    def __init__(self, model_path, target_classes, confidence_threshold):
        """
        Initializes the detector.
        Args:
            model_path (str): Path to the YOLO model weights file (e.g., 'yolov8n.pt').
            target_classes (list): A list of class IDs to detect.
            confidence_threshold (float): The minimum confidence for a detection to be considered.
        """
        self.model = YOLO(model_path)
        self.target_classes = target_classes
        self.conf_threshold = confidence_threshold

    def detect(self, frame):
        """
        Performs object detection on a single frame.
        Args:
            frame: The input video frame.
        Returns:
            An `ultralytics.engine.results.Boxes` object containing the
            filtered detections.
        """
        results = self.model(
            frame,
            classes=self.target_classes,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        return results.boxes