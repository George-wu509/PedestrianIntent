# pedestrian_intent/extractors/pose_extractor.py
import numpy as np
from .base_extractor import BaseExtractor
from ..core.structures import Pedestrian, FrameData

class PoseExtractor(BaseExtractor):
    """
    Extracts whole-body keypoints using a pre-trained MMPose model.
    
    NOTE: This is a high-level abstraction. A real implementation would use the
    MMPose Python API for inference.
    """
    def __init__(self, device: str = 'cuda'):
        print("Initializing PoseExtractor...")
        self.device = device
        self._load_model()

    def _load_model(self):
        """Placeholder for loading the MMPose model."""
        print("  - Placeholder: Loading MMPose whole-body model...")
        # from mmpose.apis import MMPoseInferencer
        # self.model = MMPoseInferencer('wholebody', device=self.device)
        self.model = "(Mock MMPose Model)"
        print("Model loaded.")

    def extract(self, pedestrian: Pedestrian, frame_data: FrameData) -> Pedestrian:
        """
        Crops the pedestrian from the full image and runs pose estimation.
        """
        # In a real implementation, you would run the model:
        # result = self.model(image_crop)
        # keypoints = result['predictions'][0][0]['keypoints']
        # keypoint_scores = result['predictions'][0][0]['keypoint_scores']
        
        # --- MOCK LOGIC START ---
        # Simulate finding 133 whole-body keypoints within the bbox
        x1, y1, x2, y2 = pedestrian.bbox
        mock_keypoints_x = np.random.uniform(x1, x2, 133)
        mock_keypoints_y = np.random.uniform(y1, y2, 133)
        mock_scores = np.random.uniform(0.8, 1.0, 133)
        
        pedestrian.keypoints = np.stack([mock_keypoints_x, mock_keypoints_y, mock_scores], axis=1)
        # --- MOCK LOGIC END ---
        
        # Also extract head bbox from facial keypoints for the GazeExtractor
        # COCO WholeBody facial keypoints are typically indices 68-132 or similar
        if pedestrian.keypoints is not None:
            facial_kps = pedestrian.keypoints[68:, :2]
            if facial_kps.shape[0] > 0:
                min_x, min_y = np.min(facial_kps, axis=0)
                max_x, max_y = np.max(facial_kps, axis=0)
                pedestrian.head_bbox = np.array([min_x, min_y, max_x, max_y])

        return pedestrian