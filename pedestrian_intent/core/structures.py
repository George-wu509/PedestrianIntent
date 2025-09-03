# pedestrian_intent/core/structures.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class DetectedObject:
    """Represents a single detected object in a frame."""
    track_id: int
    label: str
    bbox: np.ndarray  # [x1, y1, x2, y2]
    mask: np.ndarray  # Binary mask of shape (H, W)
    confidence: float
    centroid: np.ndarray = field(init=False) # [x, y]

    def __post_init__(self):
        # Calculate centroid from mask
        if self.mask is not None and np.any(self.mask):
            y_coords, x_coords = np.where(self.mask)
            self.centroid = np.array([np.mean(x_coords), np.mean(y_coords)])
        else:
            # Fallback to bbox center if mask is empty
            x1, y1, x2, y2 = self.bbox
            self.centroid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])


@dataclass
class Pedestrian(DetectedObject):
    """Extends DetectedObject with pedestrian-specific attributes."""
    keypoints: Optional[np.ndarray] = None  # Shape (N, 3) for (x, y, conf)
    gaze_vector: Optional[np.ndarray] = None # Shape (2,) for (pitch, yaw)
    head_bbox: Optional[np.ndarray] = None # Bbox for the head
    
    # This will be populated by the TrajectoryExtractor
    trajectory: List[np.ndarray] = field(default_factory=list)


@dataclass
class FrameData:
    """Holds all extracted information for a single video frame."""
    frame_id: int
    image: np.ndarray
    pedestrians: List[Pedestrian]
    scene_elements: List[DetectedObject]

@dataclass
class VideoData:
    """Stores data for an entire video, organized by track_id."""
    pedestrians: Dict[int, Pedestrian] = field(default_factory=dict)
    
    def update_frame(self, frame_data: FrameData):
        """Updates the video data with a new frame's information."""
        for ped in frame_data.pedestrians:
            if ped.track_id not in self.pedestrians:
                self.pedestrians[ped.track_id] = ped
            else:
                # Update existing pedestrian with new frame info
                # Keep trajectory history
                existing_ped = self.pedestrians[ped.track_id]
                existing_ped.bbox = ped.bbox
                existing_ped.mask = ped.mask
                existing_ped.keypoints = ped.keypoints
                existing_ped.gaze_vector = ped.gaze_vector
                existing_ped.head_bbox = ped.head_bbox
                existing_ped.trajectory.append(ped.centroid)