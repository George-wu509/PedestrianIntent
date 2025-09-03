# examples/1_feature_extraction_demo.py
import cv2
import json
import numpy as np
from pedestrian_intent.detectors import GroundedSAMDetector
from pedestrian_intent.extractors import PoseExtractor, GazeExtractor
from pedestrian_intent.core.structures import FrameData
from pedestrian_intent.utils.visualization import Visualizer

def main():
    # --- Setup ---
    config_path = "pedestrian_intent/assets/class_definitions.json"
    with open(config_path, 'r') as f:
        class_defs = json.load(f)
    
    visualizer = Visualizer(class_defs)
    detector = GroundedSAMDetector()
    pose_extractor = PoseExtractor()
    gaze_extractor = GazeExtractor()

    # --- Load Image ---
    # In a real scenario, you would load your image
    # image = cv2.imread("path/to/your/image.jpg")
    # For this demo, we create a blank image
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    print("Running feature extraction demo on a mock image.")

    # --- Processing ---
    # 1. Detect objects
    prompts = ["pedestrian", "car"]
    pedestrians, scene_elements = detector.process_frame(image, prompts)
    
    frame_data = FrameData(frame_id=0, image=image, pedestrians=pedestrians, scene_elements=scene_elements)

    # 2. Extract features for each pedestrian
    if not pedestrians:
        print("No pedestrians detected in the mock data.")
        return

    pedestrian = pedestrians[0] # Process the first detected pedestrian
    pedestrian = pose_extractor.extract(pedestrian, frame_data)
    pedestrian = gaze_extractor.extract(pedestrian, frame_data)

    # --- Visualization ---
    # Create a fresh copy of the image for drawing
    vis_image = image.copy()
    
    # Draw the extracted features
    visualizer.draw_pedestrian(vis_image, pedestrian, predictions={"crossing_intention": 0.0}) # No prediction in this demo
    visualizer.draw_scene_elements(vis_image, scene_elements)

    # --- Display ---
    cv2.imshow("Feature Extraction Demo", vis_image)
    print("Displaying results. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Demo finished.")

if __name__ == "__main__":
    main()