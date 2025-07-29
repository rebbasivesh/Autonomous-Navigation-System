import numpy as np

class CollisionAvoider:
    """
    Detects potential collisions with tracked obstacles and generates
    alternative paths to avoid them.
    """
    def __init__(self, ego_width, safe_dist, avoidance_shift):
        """
        Args:
            ego_width (float): The width of the ego vehicle in BEV pixels.
            safe_dist (float): The safe following distance in BEV pixels.
            avoidance_shift (float): The lateral distance to shift for an avoidance maneuver.
        """
        self.ego_width = ego_width
        self.safe_dist = safe_dist
        self.avoidance_shift = avoidance_shift

    def check_for_collisions(self, current_path, obstacle_bboxes_bev):
        """
        Checks if the ego vehicle's path intersects with any obstacle bounding boxes.
        Returns the first conflicting obstacle's bbox if a collision is imminent.
        
        Args:
            current_path (np.array): The ego vehicle's planned path in BEV.
            obstacle_bboxes_bev (list): A list of obstacle bounding boxes [x1, y1, x2, y2] in BEV.
        """
        # Define the ego vehicle's path corridor
        path_left_x = current_path[:, 0] - self.ego_width / 2
        path_right_x = current_path[:, 0] + self.ego_width / 2

        # Find the closest obstacle in the path
        closest_obstacle = None
        min_dist = float('inf')

        for obs_bbox in obstacle_bboxes_bev:
            obs_x1, obs_y1, obs_x2, obs_y2 = obs_bbox
            
            # Check for overlap between the path corridor and the obstacle bbox
            for i in range(len(current_path) - 1):
                path_y_start, path_y_end = current_path[i, 1], current_path[i+1, 1]
                
                # Check for y-axis (longitudinal) overlap
                y_overlap = max(path_y_start, obs_y1) < min(path_y_end, obs_y2)
                
                if y_overlap:
                    # Check for x-axis (lateral) overlap
                    x_overlap = max(path_left_x[i], obs_x1) < min(path_right_x[i], obs_x2)
                    
                    if x_overlap:
                        # Collision is imminent. Check if this is the closest obstacle.
                        dist_to_obstacle = obs_y1 # Use the top of the bbox as distance
                        if dist_to_obstacle < min_dist:
                            min_dist = dist_to_obstacle
                            closest_obstacle = obs_bbox
                        break # Move to next obstacle once overlap is found for this one
        
        return closest_obstacle # Return the closest conflicting obstacle

    def generate_avoidance_path(self, current_path, conflicting_obstacle):
        """
        Generates a new path that shifts laterally to avoid the obstacle.
        This is a simplified avoidance maneuver.
        """
        # Decide to shift left or right based on obstacle position relative to path center
        path_center_x = current_path[0, 0]
        obstacle_center_x = (conflicting_obstacle[0] + conflicting_obstacle[2]) / 2
        
        # If obstacle is to the right of our path, shift left. Otherwise, shift right.
        shift = -self.avoidance_shift if obstacle_center_x > path_center_x else self.avoidance_shift
        
        avoidance_path = current_path.copy()
        avoidance_path[:, 0] += shift
        return avoidance_path