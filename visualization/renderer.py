import cv2
import numpy as np

class PathRenderer:
    """
    Handles drawing the planned path, vehicle detections, and other
    visual information onto the video frames.
    """
    def __init__(self, perspective_transformer, ego_width_pixels):
        """
        Args:
            perspective_transformer (PerspectiveTransformer): The transformer for BEV conversions.
            ego_width_pixels (int): The width of the ego vehicle in BEV pixels.
        """
        self.transformer = perspective_transformer
        self.ego_width_pixels = ego_width_pixels

    def draw_roi(self, frame, roi_points):
        """Draws the region of interest trapezoid on the main frame."""
        cv2.polylines(frame, [np.int32(roi_points)], isClosed=True, color=(0, 255, 255), thickness=2)

    def draw_path(self, bev_frame, path, color=(0, 255, 0), thickness=2):
        """Draws a path on the bird's-eye view frame."""
        if path is not None and len(path) > 1:
            path_points = np.int32(path).reshape((-1, 1, 2))
            cv2.polylines(bev_frame, [path_points], isClosed=False, color=color, thickness=thickness)

    def draw_projected_path(self, frame, path_bev, color=(0, 255, 0)):
        """
        Projects the BEV path back onto the original frame and draws it as a filled polygon.
        """
        if path_bev is None or len(path_bev) < 2:
            return

        # Create a corridor around the path line
        path_left = path_bev.copy()
        path_left[:, 0] -= self.ego_width_pixels / 2
        path_right = path_bev.copy()
        path_right[:, 0] += self.ego_width_pixels / 2

        # Combine and project back to image space
        full_path_polygon_bev = np.vstack((path_left, path_right[::-1]))
        projected_polygon = self.transformer.transform_to_image(full_path_polygon_bev)

        if projected_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.int32(projected_polygon)], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def draw_tracked_vehicles_on_bev(self, bev_frame, tracked_objects_bev):
        """Draws tracked vehicles as points on the BEV frame."""
        for bbox in tracked_objects_bev:
            # Use the center of the bottom edge of the bbox for positioning
            x_center = int((bbox[0] + bbox[2]) / 2)
            y_bottom = int(bbox[3])
            cv2.circle(bev_frame, (x_center, y_bottom), 8, (255, 0, 0), -1)

    def draw_tracked_vehicles_on_image(self, frame, tracked_objects_image):
        """Draws bounding boxes for tracked vehicles on the main frame."""
        for track in tracked_objects_image:
            kf = track['kf']
            track_id = track['id']
            
            # Bbox in [cx, cy, w, h] format from Kalman Filter state
            cx, cy, w, h = kf.x[:4].flatten()
            
            # Convert center, width, height to x1, y1, x2, y2
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            x2, y2 = int(cx + w / 2), int(cy + h / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def combine_views(self, frame, bev_frame):
        """
        Resizes the BEV and places it on the top-right corner of the main frame.
        """
        h, w, _ = frame.shape
        bev_h, bev_w, _ = bev_frame.shape

        scale = (h / 2) / bev_h
        new_bev_w, new_bev_h = int(bev_w * scale), int(h / 2)
        bev_resized = cv2.resize(bev_frame, (new_bev_w, new_bev_h))

        frame[10:10 + new_bev_h, w - new_bev_w - 10:w - 10] = bev_resized
        return frame