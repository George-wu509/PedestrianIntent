# pedestrian_intent/pipeline.py
import cv2
import json
from tqdm import tqdm
from .core.structures import FrameData, VideoData
from .detectors import GroundedSAMDetector
from .extractors import PoseExtractor, GazeExtractor, TrajectoryExtractor
from .predictors import RuleBasedPredictor
from .utils.visualization import Visualizer

class PedestrianIntentPipeline:
    """
    The main orchestrator for the pedestrian intention prediction pipeline.
    """
    def __init__(self, config_path: str = "pedestrian_intent/assets/class_definitions.json"):
        print("Initializing Pedestrian Intent Pipeline...")
        self.detector = GroundedSAMDetector()
        
        self.video_data = VideoData() # Create a data store for the whole video
        
        self.extractors = {
            "pose": PoseExtractor(),
            "gaze": GazeExtractor(),
            "trajectory": TrajectoryExtractor(self.video_data)
        }
        self.predictor = RuleBasedPredictor()

        with open(config_path, 'r') as f:
            class_defs = json.load(f)
        
        self.prompts = [c['name'] for c in class_defs['classes']]
        self.visualizer = Visualizer(class_defs)
        print("Pipeline Initialized.")

    def process_video(self, video_path: str, output_path: str):
        """
        Processes a video file end-to-end: detection, tracking, feature extraction,
        prediction, and visualization.

        Args:
            video_path: Path to the input video file.
            output_path: Path to save the annotated output video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        # Video writer setup
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detect and segment all objects in the frame
            pedestrians, scene_elements = self.detector.process_frame(frame, self.prompts)
            
            frame_data = FrameData(frame_id, frame, pedestrians, scene_elements)

            # 2. Update video-level data store
            self.video_data.update_frame(frame_data)

            processed_pedestrians = []
            for pedestrian in frame_data.pedestrians:
                # 3. Run all feature extractors for each pedestrian
                p = self.extractors["pose"].extract(pedestrian, frame_data)
                p = self.extractors["gaze"].extract(p, frame_data)
                p = self.extractors["trajectory"].extract(p, frame_data)
                
                # 4. Predict intention
                predictions = self.predictor.predict(p, frame_data)
                
                # 5. Visualize results
                self.visualizer.draw_pedestrian(frame, p, predictions)

            self.visualizer.draw_scene_elements(frame, frame_data.scene_elements)
            out.write(frame)
            frame_id += 1
        
        cap.release()
        out.release()
        print(f"Processing complete. Annotated video saved to: {output_path}")