# pedestrian_intent/extractors/trajectory_extractor.py
from .base_extractor import BaseExtractor
from ..core.structures import Pedestrian, FrameData, VideoData

class TrajectoryExtractor(BaseExtractor):
    """
    Extracts the trajectory of a pedestrian over time.
    
    This extractor's main job is to ensure the trajectory data from the
    global VideoData object is correctly referenced in the current frame's
    pedestrian object.
    """
    def __init__(self, video_data: VideoData):
        print("Initializing TrajectoryExtractor...")
        self.video_data = video_data

    def extract(self, pedestrian: Pedestrian, frame_data: FrameData) -> Pedestrian:
        """
        Retrieves the historical trajectory for the pedestrian from the video data store.
        """
        if pedestrian.track_id in self.video_data.pedestrians:
            # The trajectory is already being built in the VideoData object.
            # We just copy the reference here for the current frame's context.
            pedestrian.trajectory = self.video_data.pedestrians[pedestrian.track_id].trajectory
        
        return pedestrian