# Enhanced Pedestrian Intention Prediction based on Dynamic Behavior Decoding



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

## Core Technology Stack
This project integrates several cutting-edge models and libraries to achieve its capabilities.


| Category          | Technology                 | Purpose                                                   |
|-------------------|----------------------------|-----------------------------------------------------------|
| Foundation Models | Grounding DINO             | Zero-Shot Object Detection via Text Prompts               |
|                   | Segment Anything (SAM/SAM2)| High-Quality Object Segmentation & Video Tracking         |
| Feature Extractors| MMPose                     | Whole-Body (133 keypoints) Pose Estimation                |
|                   | ETH-XGaze (or similar)     | Gaze and Head Pose Estimation                             |
| Core Libraries    | Python 3.9+                | Core Programming Language                                 |
|                   | PyTorch                    | Deep Learning Framework                                   |
|                   | Transformers (Hugging Face)| For easy access to models like Grounding DINO             |
|                   | OpenCV                     | Video/Image Processing & Visualization                    |
| Project Management| Poetry                     | Dependency and Environment Management                     |

## Project Structure

The repository is organized in a standard Python library structure for clarity and scalability.

```bash
PedestrianIntent/
├── pyproject.toml              # Project metadata and dependencies for Poetry
├── README.md                   # This file
├── examples/                   # Example scripts showing how to use the library
│   ├── 1_feature_extraction_demo.py
│   └── 2_intention_prediction_pipeline.py
├── pedestrian_intent/          # The main library source code
│   ├── assets/                 # Static assets like class definitions
│   ├── core/                   # Core data structures (e.g., Pedestrian, FrameData)
│   ├── detectors/              # Zero-shot object detection and tracking modules
│   ├── extractors/             # Modules for pose, gaze, and trajectory extraction
│   ├── predictors/             # Intention prediction models (e.g., rule-based)
│   ├── utils/                  # Helper utilities, including visualization tools
│   └── pipeline.py             # The main orchestrator that connects all modules
└── scripts/
    └── download_models.sh      # Script to download pre-trained model weights
```


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

You can run the full end-to-end pipeline on a video file with just a few lines of code. The following example processes a video and saves a new version with all visualizations overlaid.

Place your input video (e.g., my_video.mp4) in the project's root directory.

```python
# Found in: examples/2_intention_prediction_pipeline.py

from pedestrian_intent.pipeline import PedestrianIntentPipeline

def main():
    # 1. Initialize the full end-to-end pipeline.
    #    This will load all the necessary models into memory.
    print("Initializing the Pedestrian Intent Pipeline...")
    pipeline = PedestrianIntentPipeline()

    # 2. Process a video and save the annotated output.
    #    The pipeline handles frame-by-frame detection, tracking, feature
    #    extraction, prediction, and visualization.
    print("Starting video processing...")
    pipeline.process_video(
        video_path="my_video.mp4",         # Your input video file
        output_path="annotated_output.mp4" # Where to save the result
    )
    print("Processing complete. Check the output file: annotated_output.mp4")

if __name__ == "__main__":
    main()
```

To run the example:
```bash
# Make sure your poetry environment is active
poetry shell

# Run the quick start script
python examples/2_intention_prediction_pipeline.py
```
