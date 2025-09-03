#!/bin/bash

# This script provides guidance on downloading the necessary pre-trained models.
# Actual download commands might need to be sourced from the models' official pages.

echo "--- PedestrianIntent Model Download Script ---"

# Create a directory to store models
mkdir -p models
cd models

# --- Grounding DINO ---
echo "[1/4] Downloading Grounding DINO models..."
# In a real scenario, you'd use wget or git clone.
# Example from the official repository:
# wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
echo "  -> Placeholder: Please download Grounding DINO weights from its official GitHub repository."
echo "     https://github.com/IDEA-Research/GroundingDINO"


# --- Segment Anything Model (SAM) ---
echo "[2/4] Downloading Segment Anything Model (SAM) weights..."
# Example:
# wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
echo "  -> Placeholder: Please download SAM weights from the official Meta AI website."
echo "     https://github.com/facebookresearch/segment-anything"
echo "     (Note: We conceptualize SAM2; use the latest available powerful SAM model.)"


# --- MMPose ---
echo "[3/4] Downloading MMPose models..."
echo "  -> MMPose models are typically downloaded automatically via their API."
echo "     Ensure you have followed MMPose installation instructions."
echo "     Models will be cached in your user directory upon first use."


# --- Gaze Estimation Model (e.g., from ETH-XGaze) ---
echo "[4/4] Downloading Gaze Estimation models..."
echo "  -> Placeholder: Please download the gaze model weights from its official repository."
echo "     Example for ETH-XGaze: https://github.com/xucong-zhang/ETH-XGaze"


cd ..
echo "--- Download script finished. ---"
echo "Please ensure you have manually downloaded any placeholder models."