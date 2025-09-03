# pedestrian_intent/__init__.py
"""
Pedestrian Intent Prediction Library
==================================

A library for pedestrian crossing intention prediction using foundation models.
"""
__version__ = "0.1.0"

from .pipeline import PedestrianIntentPipeline
from .core.structures import FrameData, Pedestrian