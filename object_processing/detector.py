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
            A list of `ultralytics.engine.results.Boxes` objects for
            the detected vehicles.
        """
        results = self.model(frame, verbose=False)[0]
        detections = [res for res in results.boxes if res.cls[0] in self.target_classes and res.conf[0] > self.conf_threshold]
        return detections