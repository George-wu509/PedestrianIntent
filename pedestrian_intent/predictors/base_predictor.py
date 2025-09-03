# pedestrian_intent/predictors/base_predictor.py
from abc import ABC, abstractmethod
from typing import Dict
from ..core.structures import Pedestrian, FrameData

class BasePredictor(ABC):
    """Abstract base class for all intention predictors."""

    @abstractmethod
    def predict(self, pedestrian: Pedestrian, frame_data: FrameData) -> Dict[str, float]:
        """
        Predicts the intention of a single pedestrian.

        Args:
            pedestrian: The pedestrian object with all features extracted.
            frame_data: The scene context for the current frame.

        Returns:
            A dictionary containing prediction scores, e.g., {"crossing_intention": 0.8}.
        """
        pass