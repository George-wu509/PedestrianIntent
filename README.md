# Pedestrian Intent Prediction Library

![Demo GIF](https://your-link-to-a-cool-demo-gif.com/demo.gif)

**`pedestrian-intent`** is a powerful Python library designed for real-time pedestrian crossing intention prediction from video streams. It leverages the power of cutting-edge, pre-trained foundation models like **Grounding DINO** and **SAM2** to perform zero-shot detection, segmentation, and tracking, eliminating the need for traditional dataset annotation and model training for feature extraction.

---

## Features

-   **Zero-Shot Detection**: Uses Grounding DINO to detect pedestrians, vehicles, crosswalks, and other scene elements using simple text prompts.
-   **Precise Segmentation & Tracking**: Employs the Segment Anything Model (SAM/SAM2) to get high-quality masks of detected objects and track them across video frames.
-   **Multi-Modal Feature Extraction**:
    -   **Pose Estimation**: Integrates with MMPose for whole-body keypoint detection.
    -   **Gaze Estimation**: Uses pre-trained models like ETH-XGaze to determine where the pedestrian is looking.
    -   **Trajectory Analysis**: Calculates object trajectories from tracked masks.
-   **Modular & Extensible**: A clean, object-oriented structure makes it easy to add new detectors, feature extractors, or prediction models.
-   **Rich Visualization**: Comes with built-in utilities to visualize all extracted features (skeletons, masks, gaze vectors, trajectories) on the video.

## How It Works

This library operates on a new paradigm: orchestrating powerful foundation models instead of training custom ones.

1.  **Detect & Track**: `GroundedSAMDetector` takes text prompts (e.g., "pedestrian", "car") and processes a video to find and track these objects frame by frame.
2.  **Extract Attributes**: For each tracked pedestrian, a series of `Extractor` modules are run to gather fine-grained details:
    -   `PoseExtractor` finds body keypoints.
    -   `GazeExtractor` determines head orientation and gaze.
    -   `TrajectoryExtractor` computes the path of movement.
3.  **Predict Intention**: A `Predictor` module takes the rich, multi-modal feature set and applies logic (e.g., a rule-based engine or a simple temporal model) to predict the probability of a crossing intention.

## Installation

This project is managed with [Poetry](https://python-poetry.org/).

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/PedestrianIntent.git](https://github.com/your-username/PedestrianIntent.git)
    cd PedestrianIntent
    ```

2.  **Install dependencies using Poetry:**
    (If you don't have Poetry, [install it first](https://python-poetry.org/docs/#installation).)
    ```bash
    poetry install
    ```

3.  **Install MMLab Libraries (MMPose/MMCV):**
    These libraries require specific installation steps depending on your PyTorch and CUDA versions. Please follow the [official OpenMMLab guide](https://mmpose.readthedocs.io/en/latest/installation.html).

4.  **Download Pre-trained Models:**
    Run the provided script to download the necessary model weights for Grounding DINO, SAM, MMPose, etc.
    ```bash
    bash scripts/download_models.sh
    ```

## Quickstart

Check out the `examples/` directory for detailed usage. Here's a simple example of running the full pipeline on a video:

```python
# examples/2_intention_prediction_pipeline.py

from pedestrian_intent.pipeline import PedestrianIntentPipeline

def main():
    # Initialize the full pipeline
    pipeline = PedestrianIntentPipeline()

    # Process a video and save the annotated output
    pipeline.process_video(
        video_path="path/to/your/input_video.mp4",
        output_path="path/to/your/output_video.mp4"
    )

if __name__ == "__main__":
    main()

To-Do & Future Work
[ ] Implement a Transformer-based predictor for more advanced temporal reasoning.

[ ] Add support for real-time stream processing (e.g., from a webcam).

[ ] Integrate with large vision-language models (VLMs) for the final prediction step.

License
This project is licensed under the MIT License. See the LICENSE file for details.