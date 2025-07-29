import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = bbox1
    x1_p, y1_p, x2_p, y2_p = bbox2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class EKFTracker:
    """
    Tracks objects using a Kalman Filter and Hungarian algorithm for assignment.
    Note: While named EKFTracker as per the request, this implementation uses a
    standard Kalman Filter with a linear constant velocity model, which is
    sufficient and more common for this type of bounding box tracking.
    """
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.tracks = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age # Max frames to keep a track without a detection

    def _create_kalman_filter(self):
        """Creates a Kalman Filter for a new track."""
        kf = KalmanFilter(dim_x=6, dim_z=4)
        # State vector [x, y, w, h, vx, vy] (center_x, center_y, width, height, vel_x, vel_y)
        # State Transition Matrix (F)
        kf.F = np.array([[1,0,0,0,1,0],
                         [0,1,0,0,0,1],
                         [0,0,1,0,0,0],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])
        # Measurement Matrix (H) - we only measure position and size
        kf.H = np.array([[1,0,0,0,0,0],
                         [0,1,0,0,0,0],
                         [0,0,1,0,0,0],
                         [0,0,0,1,0,0]])
        # Measurement Noise Covariance (R) - uncertainty in the detection measurement
        kf.R *= 10.
        # Process Noise Covariance (Q) - uncertainty in the motion model
        kf.Q[4:,4:] *= 0.01
        kf.Q[:4,:4] *= 0.01
        # Initial State Covariance (P) - initial uncertainty of the state
        kf.P *= 10.
        return kf

    def update(self, detections):
        # 1. Predict the next state for all existing tracks
        for track in self.tracks:
            track['kf'].predict()
            track['age'] += 1

        # 2. Associate detections with tracks using IoU and Hungarian algorithm
        unmatched_detections_indices = set(range(len(detections)))
        matched_track_indices = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            # Get predicted track bboxes in [cx, cy, w, h]
            track_pred_bboxes_xywh = np.array([t['kf'].x[:4].flatten() for t in self.tracks])
            
            # Convert to [x1, y1, x2, y2] for IoU calculation
            track_pred_bboxes_xyxy = track_pred_bboxes_xywh.copy()
            track_pred_bboxes_xyxy[:, 0] -= track_pred_bboxes_xyxy[:, 2] / 2
            track_pred_bboxes_xyxy[:, 1] -= track_pred_bboxes_xyxy[:, 3] / 2
            track_pred_bboxes_xyxy[:, 2] += track_pred_bboxes_xyxy[:, 0]
            track_pred_bboxes_xyxy[:, 3] += track_pred_bboxes_xyxy[:, 1]

            det_bboxes_xywh = np.array([d.xywh[0] for d in detections])
            det_bboxes_xyxy = np.array([d.xyxy[0] for d in detections])

            iou_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
            for d, det_box in enumerate(det_bboxes_xyxy):
                for t, trk_box in enumerate(track_pred_bboxes_xyxy):
                    iou_matrix[d, t] = iou(det_box, trk_box)

            # Use Hungarian algorithm for optimal assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix) # Maximize IoU
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    # 3. Update matched tracks with new detection info
                    detection_measurement = det_bboxes_xywh[r] # [cx, cy, w, h]
                    self.tracks[c]['kf'].update(detection_measurement)
                    self.tracks[c]['age'] = 0
                    self.tracks[c]['hits'] += 1
                    matched_track_indices.add(c)
                    if r in unmatched_detections_indices:
                        unmatched_detections_indices.remove(r)
        
        # 4. Create new tracks for unmatched detections
        for i in unmatched_detections_indices:
            det = detections[i]
            kf = self._create_kalman_filter()
            kf.x[:4] = det.xywh[0].reshape(4, 1)
            self.tracks.append({'id': self.next_id, 'kf': kf, 'age': 0, 'hits': 1})
            self.next_id += 1

        # 5. Remove old tracks that haven't been seen for a while
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        # Return active tracks (those that have been updated recently and have some history)
        return [t for t in self.tracks if t['hits'] > 3 and t['age'] == 0]