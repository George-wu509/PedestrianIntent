# pedestrian_intent/utils/visualization.py
import cv2
import numpy as np
import random
from typing import Dict, List
from ..core.structures import Pedestrian, DetectedObject

class Visualizer:
    """A class to handle all visualization tasks."""
    def __init__(self, class_definitions: Dict):
        self.colors = {c['name']: tuple(c['color_rgb']) for c in class_definitions['classes']}
        # MMPose COCO-WholeBody connections
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Body
            (5, 11), (6, 12), (11, 12), # Hips
            (11, 13), (13, 15), (12, 14), (14, 16) # Legs
        ]

    def _get_color(self, label: str) -> tuple:
        return self.colors.get(label, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    def draw_mask(self, image: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
        """Draws a semi-transparent mask on the image."""
        overlay = image.copy()
        overlay[mask] = color
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def draw_skeleton(self, image: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3):
        """Draws skeleton connections based on keypoints."""
        for i, j in self.skeleton_connections:
            if keypoints.shape[0] > max(i, j):
                pt1 = keypoints[i]
                pt2 = keypoints[j]
                if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                    cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
    
    def draw_gaze(self, image: np.ndarray, pedestrian: Pedestrian):
        """Draws the gaze vector as an arrow."""
        if pedestrian.gaze_vector is None or pedestrian.head_bbox is None:
            return
            
        pitch, yaw = pedestrian.gaze_vector
        center_x = int((pedestrian.head_bbox[0] + pedestrian.head_bbox[2]) / 2)
        center_y = int((pedestrian.head_bbox[1] + pedestrian.head_bbox[3]) / 2)
        
        length = 100
        dx = length * np.sin(yaw) * np.cos(pitch)
        dy = length * -np.sin(pitch)
        
        end_point = (int(center_x + dx), int(center_y + dy))
        cv2.arrowedLine(image, (center_x, center_y), end_point, (255, 0, 255), 3)

    def draw_trajectory(self, image: np.ndarray, trajectory: List[np.ndarray]):
        """Draws the trajectory as a polyline."""
        if len(trajectory) < 2:
            return
        
        pts = np.array(trajectory, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=(255, 255, 0), thickness=2)

    def draw_pedestrian(self, image: np.ndarray, pedestrian: Pedestrian, predictions: Dict):
        """Draws all information for a single pedestrian."""
        color = self._get_color(pedestrian.label)
        
        # Draw mask
        image = self.draw_mask(image, pedestrian.mask, color)
        
        # Draw bounding box
        x1, y1, x2, y2 = pedestrian.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw skeleton
        if pedestrian.keypoints is not None:
            self.draw_skeleton(image, pedestrian.keypoints)
            
        # Draw gaze
        self.draw_gaze(image, pedestrian)
        
        # Draw trajectory
        self.draw_trajectory(image, pedestrian.trajectory)
        
        # Write text (ID and intention score)
        intention = predictions.get("crossing_intention", 0.0)
        label_text = f"ID: {pedestrian.track_id} | Intent: {intention:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_scene_elements(self, image: np.ndarray, elements: List[DetectedObject]):
        """Draws all other scene elements."""
        for element in elements:
            color = self._get_color(element.label)
            image = self.draw_mask(image, element.mask, color, alpha=0.3)
            x1, y1, x2, y2 = element.bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, element.label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)