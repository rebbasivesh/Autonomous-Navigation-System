import cv2
import numpy as np
import config

points = []

def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse clicks to select points for perspective transformation.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)} added: ({x}, {y})")
        if len(points) == 4:
            print("\nFour points selected. Press 'q' to quit.")
            print("Copy the following line into your config.py for SRC_POINTS:\n")
            print(f"SRC_POINTS = np.float32({points})")

def main():
    """
    Main function to run the calibration tool.
    """
    cap = cv2.VideoCapture(config.VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.VIDEO_INPUT_PATH}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    cv2.namedWindow("Calibration - Click 4 points (TL, TR, BR, BL) then press 'q'")
    cv2.setMouseCallback("Calibration - Click 4 points (TL, TR, BR, BL) then press 'q'", mouse_callback)

    while True:
        display_frame = frame.copy()
        for point in points:
            cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
        cv2.imshow("Calibration - Click 4 points (TL, TR, BR, BL) then press 'q'", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()