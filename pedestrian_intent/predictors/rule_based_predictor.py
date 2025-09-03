# pedestrian_intent/predictors/rule_based_predictor.py
import numpy as np
from typing import Dict, Optional
from .base_predictor import BasePredictor
from ..core.structures import Pedestrian, FrameData, DetectedObject

class RuleBasedPredictor(BasePredictor):
    """
    Predicts crossing intention based on a simple, interpretable set of rules.
    """
    def __init__(self, distance_threshold: float = 50.0, yaw_threshold: float = 0.5, speed_threshold: float = 0.2):
        print("Initializing RuleBasedPredictor...")
        self.distance_threshold = distance_threshold  # pixels
        self.yaw_threshold = yaw_threshold            # radians (~30 degrees)
        self.speed_threshold = speed_threshold        # pixels per frame

    def _get_closest_element(self, pedestrian: Pedestrian, elements: list[DetectedObject]) -> Optional[DetectedObject]:
        if not elements:
            return None
        
        ped_center = pedestrian.centroid
        closest_element = min(elements, key=lambda e: np.linalg.norm(ped_center - e.centroid))
        return closest_element

    def predict(self, pedestrian: Pedestrian, frame_data: FrameData) -> Dict[str, float]:
        """Applies a set of rules to estimate crossing intention."""
        score = 0.0
        weights = {"distance": 0.4, "gaze": 0.4, "movement": 0.2}

        # Rule 1: Proximity to road or crosswalk
        roads = [e for e in frame_data.scene_elements if e.label == 'road']
        crosswalks = [e for e in frame_data.scene_elements if e.label == 'crosswalk']
        
        closest_road = self._get_closest_element(pedestrian, roads)
        if closest_road:
            dist = np.linalg.norm(pedestrian.centroid - closest_road.centroid)
            if dist < self.distance_threshold:
                score += weights["distance"]

        # Rule 2: Gaze direction (is pedestrian looking at the road?)
        if pedestrian.gaze_vector is not None:
            yaw = pedestrian.gaze_vector[1]
            # Assuming yaw=0 is straight ahead, positive is right, negative is left
            if abs(yaw) < self.yaw_threshold: # Looking relatively straight
                score += weights["gaze"]
        
        # Rule 3: Movement direction and speed
        if len(pedestrian.trajectory) > 5:
            # Calculate recent velocity
            p1 = pedestrian.trajectory[-1]
            p0 = pedestrian.trajectory[-5]
            velocity = (p1 - p0) / 4.0
            speed = np.linalg.norm(velocity)

            if speed > self.speed_threshold:
                # Simple check: is movement direction towards the road?
                if closest_road:
                    direction_to_road = closest_road.centroid - p1
                    # Check if velocity vector is aligned with direction to road
                    cosine_similarity = np.dot(velocity, direction_to_road) / (np.linalg.norm(velocity) * np.linalg.norm(direction_to_road))
                    if cosine_similarity > 0.5: # Roughly aligned
                        score += weights["movement"]

        return {"crossing_intention": min(1.0, score)}