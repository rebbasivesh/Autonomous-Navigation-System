import cv2
import numpy as np

class PerspectiveTransformer:
    """
    Handles the transformation of an image from the camera's perspective
    to a bird's-eye view (Inverse Perspective Mapping - IPM).
    """
    def __init__(self, src_points, dst_points):
        """
        Initializes the transformer with source and destination points.
        Args:
            src_points (np.float32): 4 points in the source image (camera view).
            dst_points (np.float32): 4 corresponding points in the destination image (BEV).
        """
        # Calculate the perspective transform matrix (M) to warp from camera to BEV
        self.M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Calculate the inverse perspective transform matrix (M_inv) to warp from BEV to camera
        self.M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    def transform_to_bev(self, image_or_points):
        """
        Transforms an image or a set of points to the bird's-eye view.
        Args:
            image_or_points: The input image or a numpy array of points (N, 1, 2).
        Returns:
            The transformed image or points.
        """
        # Ensure the input is a numpy array with the correct shape
        points = np.array(image_or_points, dtype=np.float32)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.M)

    def transform_to_image(self, image_or_points):
        """
        Transforms a bird's-eye view image or points back to the camera's perspective.
        Args:
            image_or_points: The input BEV image or a numpy array of points (N, 1, 2).
        Returns:
            The transformed image or points.
        """
        # Ensure the input is a numpy array with the correct shape
        points = np.array(image_or_points, dtype=np.float32)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.M_inv)
