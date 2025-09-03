# pedestrian_intent/extractors/gaze_extractor.py
import numpy as np
from .base_extractor import BaseExtractor
from ..core.structures import Pedestrian, FrameData
import cv2

class GazeExtractor(BaseExtractor):
    """
    Extracts gaze direction using a pre-trained model like ETH-XGaze.
    
    NOTE: This is a high-level abstraction.
    """
    def __init__(self, device: str = 'cuda'):
        print("Initializing GazeExtractor...")
        self.device = device
        self._load_model()

    def _load_model(self):
        """Placeholder for loading the Gaze Estimation model."""
        print("  - Placeholder: Loading ETH-XGaze model...")
        self.model = "(Mock Gaze Model)"
        print("Model loaded.")

    def extract(self, pedestrian: Pedestrian, frame_data: FrameData) -> Pedestrian:
        """
        Crops the head region and runs gaze estimation.
        """
        if pedestrian.head_bbox is None:
            return pedestrian

        x1, y1, x2, y2 = pedestrian.head_bbox.astype(int)
        head_image = frame_data.image[y1:y2, x1:x2]

        if head_image.size == 0:
            return pedestrian

        # Preprocess for model (e.g., resize to 224x224)
        # processed_head = cv2.resize(head_image, (224, 224))
        
        # In a real implementation, you would run the model:
        # pitch, yaw = self.model.predict(processed_head)
        
        # --- MOCK LOGIC START ---
        # Simulate a random gaze vector (pitch, yaw) in radians
        mock_pitch = np.random.uniform(-np.pi/4, np.pi/4)
        mock_yaw = np.random.uniform(-np.pi/2, np.pi/2)
        pedestrian.gaze_vector = np.array([mock_pitch, mock_yaw])
        # --- MOCK LOGIC END ---
        
        return pedestrian