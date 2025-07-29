import cv2
import numpy as np

import config
from object_processing.detector import Detector
from object_processing.tracker import EKFTracker
from visualization.perspective import PerspectiveTransformer
from visualization.renderer import PathRenderer
from navigation.path_planner import PathPlanner
from navigation.collision_avoider import CollisionAvoider

def main():
    """
    Main function to run the autonomous navigation simulation.
    """
    # --- Initialization ---
    cap = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.VIDEO_INPUT_PATH}")
        return

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        config.VIDEO_OUTPUT_PATH, fourcc, 30.0,
        (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
    )

    # Initialize components
    detector = Detector(
        model_path=config.YOLO_MODEL_PATH,
        target_classes=config.YOLO_TARGET_CLASSES,
        confidence_threshold=config.YOLO_CONFIDENCE_THRESHOLD
    )
    tracker = EKFTracker()
    transformer = PerspectiveTransformer(config.SRC_POINTS, config.DST_POINTS)
    path_planner = PathPlanner(config.DST_WIDTH, config.DST_HEIGHT, config.PLANNING_HORIZON)
    collision_avoider = CollisionAvoider(
        ego_width=config.EGO_VEHICLE_WIDTH_PIXELS,
        safe_dist=config.SAFE_DISTANCE,
        avoidance_shift=config.AVOIDANCE_SHIFT
    )
    renderer = PathRenderer(transformer, config.EGO_VEHICLE_WIDTH_PIXELS)

    # --- Main Processing Loop ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # 1. Object Detection
        detections = detector.detect(frame)

        # 2. Object Tracking
        active_tracks = tracker.update(detections)

        # 3. Perspective Transformation of tracked objects
        tracked_bboxes_bev = []
        if active_tracks:
            # Get the bottom center point of each bounding box in the image view
            image_points = np.array([
                [(t['kf'].x[0]), (t['kf'].x[1] + t['kf'].x[3] / 2)] for t in active_tracks
            ])
            # Transform these points to BEV
            bev_points = transformer.transform_to_bev(image_points)
            
            # Create approximate bounding boxes in BEV for collision checking
            for i, track in enumerate(active_tracks):
                cx, cy = bev_points[i][0]
                # Use a fixed pixel width for vehicles in BEV for simplicity
                w = config.EGO_VEHICLE_WIDTH_PIXELS 
                h = config.EGO_VEHICLE_LENGTH / config.METERS_PER_PIXEL_Y
                tracked_bboxes_bev.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # 4. Path Planning and Collision Avoidance
        current_path = path_planner.get_path()
        conflicting_obstacle = collision_avoider.check_for_collisions(current_path, tracked_bboxes_bev)
        
        if conflicting_obstacle:
            final_path = collision_avoider.generate_avoidance_path(current_path, conflicting_obstacle)
        else:
            final_path = current_path

        # 5. Visualization
        bev_frame = np.zeros((config.DST_HEIGHT, config.DST_WIDTH, 3), dtype=np.uint8)
        renderer.draw_roi(frame, config.SRC_POINTS)
        renderer.draw_path(bev_frame, final_path, color=(0, 255, 0))
        renderer.draw_tracked_vehicles_on_bev(bev_frame, tracked_bboxes_bev)
        renderer.draw_projected_path(frame, final_path, color=(0, 255, 0))
        renderer.draw_tracked_vehicles_on_image(frame, active_tracks)
        combined_frame = renderer.combine_views(frame, bev_frame)

        out.write(combined_frame)
        cv2.imshow('Autonomous Navigation', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to output_video.avi")

if __name__ == '__main__':
    main()