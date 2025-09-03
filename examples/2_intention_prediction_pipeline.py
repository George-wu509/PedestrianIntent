# examples/2_intention_prediction_pipeline.py
import os
import cv2
import numpy as np
from pedestrian_intent.pipeline import PedestrianIntentPipeline

def create_dummy_video(path="dummy_video.mp4", width=1280, height=720, frames=100):
    """Creates a simple dummy video for demonstration purposes."""
    if os.path.exists(path):
        print(f"Dummy video '{path}' already exists.")
        return

    print(f"Creating a dummy video at '{path}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = f"Frame: {i+1}/{frames}"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print("Dummy video created.")

def main():
    # Create a dummy video to run the pipeline on
    input_video = "dummy_input.mp4"
    output_video = "annotated_output.mp4"
    create_dummy_video(input_video)

    print("\nStarting the full intention prediction pipeline...")
    # Initialize the full pipeline
    pipeline = PedestrianIntentPipeline()

    # Process the video and save the annotated output
    pipeline.process_video(
        video_path=input_video,
        output_path=output_video
    )
    
    print(f"\nPipeline finished. Check the output file: '{output_video}'")

if __name__ == "__main__":
    main()