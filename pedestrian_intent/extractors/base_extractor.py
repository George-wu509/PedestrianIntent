# pedestrian_intent/extractors/base_extractor.py
from abc import ABC, abstractmethod
from ..core.structures import Pedestrian, FrameData

class BaseExtractor(ABC):
    """Abstract base class for all feature extractors."""
    
    @abstractmethod
    def extract(self, pedestrian: Pedestrian, frame_data: FrameData) -> Pedestrian:
        """
        Extracts a specific feature for a pedestrian and returns the updated object.
        
        Args:
            pedestrian: The Pedestrian object to be updated.
            frame_data: The full data of the current frame, for context.
            
        Returns:
            The updated Pedestrian object with the new feature.
        """
        pass