import numpy as np

class PathPlanner:
    """
    Manages the vehicle's intended path.
    For this simulation, it generates a static, predefined "global" path.
    """
    def __init__(self, bev_width, bev_height, horizon):
        """
        Args:
            bev_width (int): Width of the bird's-eye view in pixels.
            bev_height (int): Height of the bird's-eye view in pixels.
            horizon (int): How far ahead the path should extend, in pixels.
        """
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.horizon = horizon
        # Create a simple straight path down the center of the BEV view
        self.global_path = np.array([
            [self.bev_width / 2, y] for y in np.linspace(0, self.horizon, 50)
        ], dtype=np.float32)

    def get_path(self):
        return self.global_path.copy()