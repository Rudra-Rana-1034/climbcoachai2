# Climb Coach AI

    Climb Coach AI is a Streamlit app for analyzing climbing movement from video using pose estimation, joint-angle calculation, center of gravity tracking, and contact-frame comparison.

    It is designed to help climbers review technique, compare attempts, and visualize movement more clearly with pose overlays and side-by-side analysis.


## Features

- **Mode 1: Pose Overlay**  
  Single-video analysis with a skeleton overlay, joint angles, and center of gravity.

- **Mode 2: Side-by-Side Comparison**  
  Compare two climbing attempts from the start of the movement to the contact frame.

- **Mode 3: Overlapped Contact Frames**  
  Blend two contact frames together and compare key joint-angle metrics.

- **Contact Detection**  
  Uses velocity-based heuristics to estimate the contact frame, with manual override options.

- **Pose Model Selection**  
  Choose between Nano, Small, Medium, Large, and Extra-large YOLOv8-Pose models depending on speed vs. accuracy.

- **Temporal Smoothing**  
  Optional smoothing reduces keypoint jitter for more stable overlays.

- **Inference Resolution Control**  
  Choose standard or high inference resolution for better keypoint precision on high-quality videos.

- **Video Quality Guidance**  
  The app checks video resolution and FPS and warns when the input is too low for reliable pose detection.

- **Video Context Inputs**  
  Optional camera-angle and wall-angle inputs help guide future improvements and user feedback.

## Architecture

The processing pipeline works like this:

1. The user uploads one or two climbing videos.
2. YOLO8-based pose estimation extracts body landmarks frame by frame.
3. The app converts pose data into joint angles and center of gravity values.
4. Contact detection estimates the most useful frame for comparison.
5. The app renders one of three display modes:
   - Pose overlay
   - Side-by-side comparison
   - Overlapped contact-frame comparison

## Accuracy Improvements

After testing, I added several improvements to make the overlay more stable and reliable:

- **Video quality checks** to validate resolution and FPS before processing.
- **Camera-angle guidance** so users are encouraged to record from a side view.
- **Temporal smoothing** using OneEuroFilter to reduce jitter.
- **Higher inference resolution** for sharper keypoint localization on better videos.
- **Confidence-aware handling** for low-confidence keypoints.

These changes were added to improve pose consistency, reduce visual noise, and make the app more useful for real climbing footage.

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Ultralytics YOLOv8-Pose
- OneEuroFilter

## Requirements

- Python 3.9 or higher
- Python 3.10 recommended

## Setup

### 1. Clone the repository

Use **HTTPS** (recommended if you have not set up SSH keys with GitHub):

```bash
git clone https://github.com/Rudra-Rana-1034/climbcoachai2.git
cd climbcoachai2
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

If that does not work, try:

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to use

1. Open the app.
2. Choose an analysis mode from the sidebar.
3. Upload one video for overlay analysis, or two videos for comparison modes.
4. Adjust the pose model, inference resolution, and smoothing options if needed.
5. Review the overlay, contact frame, and metrics output.

## Tips for better results

- Record from the **side view** when possible.
- Use good lighting.
- Keep the climber’s full body in frame.
- Prefer 720p or 1080p video.
- Use 24 FPS or higher.
- Increase inference resolution for higher-quality videos if you want better keypoint precision.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| `ModuleNotFoundError: No module named 'src'` | Run Streamlit from the project folder (the directory that contains `app.py` and the `src` package). The app adds that folder to Python’s path automatically; if the error persists, use `python -m streamlit run app.py` with your venv activated. |
| Missing module errors | Reinstall dependencies with `pip install -r requirements.txt` |
| Model download fails | Ensure internet access on first run |
| Processing is slow | Use a smaller pose model or standard inference resolution |
| Overlay is jittery | Enable smoothing |
| No pose detected | Make sure the climber is fully visible and well lit |

## Project goal

This project was built to help climbers understand their movement better.
# Climb Coach AI

    Climb Coach AI is a Streamlit app for analyzing climbing movement from video using pose estimation, joint-angle calculation, center of gravity tracking, and contact-frame comparison.

    It is designed to help climbers review technique, compare attempts, and visualize movement more clearly with pose overlays and side-by-side analysis.


## Features

- **Mode 1: Pose Overlay**  
  Single-video analysis with a skeleton overlay, joint angles, and center of gravity.

- **Mode 2: Side-by-Side Comparison**  
  Compare two climbing attempts from the start of the movement to the contact frame.

- **Mode 3: Overlapped Contact Frames**  
  Blend two contact frames together and compare key joint-angle metrics.

- **Contact Detection**  
  Uses velocity-based heuristics to estimate the contact frame, with manual override options.

- **Pose Model Selection**  
  Choose between Nano, Small, Medium, Large, and Extra-large YOLOv8-Pose models depending on speed vs. accuracy.

- **Temporal Smoothing**  
  Optional smoothing reduces keypoint jitter for more stable overlays.

- **Inference Resolution Control**  
  Choose standard or high inference resolution for better keypoint precision on high-quality videos.

- **Video Quality Guidance**  
  The app checks video resolution and FPS and warns when the input is too low for reliable pose detection.

- **Video Context Inputs**  
  Optional camera-angle and wall-angle inputs help guide future improvements and user feedback.

## Architecture

The processing pipeline works like this:

1. The user uploads one or two climbing videos.
2. YOLO8-based pose estimation extracts body landmarks frame by frame.
3. The app converts pose data into joint angles and center of gravity values.
4. Contact detection estimates the most useful frame for comparison.
5. The app renders one of three display modes:
   - Pose overlay
   - Side-by-side comparison
   - Overlapped contact-frame comparison

## Accuracy Improvements

After testing, I added several improvements to make the overlay more stable and reliable:

- **Video quality checks** to validate resolution and FPS before processing.
- **Camera-angle guidance** so users are encouraged to record from a side view.
- **Temporal smoothing** using OneEuroFilter to reduce jitter.
- **Higher inference resolution** for sharper keypoint localization on better videos.
- **Confidence-aware handling** for low-confidence keypoints.

These changes were added to improve pose consistency, reduce visual noise, and make the app more useful for real climbing footage.

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Ultralytics YOLOv8-Pose
- OneEuroFilter

## Requirements

- Python 3.9 or higher
- Python 3.10 recommended

## Setup

### 1. Clone the repository

Use **HTTPS** (recommended if you have not set up SSH keys with GitHub):

```bash
git clone https://github.com/Rudra-Rana-1034/climbcoachai2.git
cd climbcoachai2
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

If that does not work, try:

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## How to use

1. Open the app.
2. Choose an analysis mode from the sidebar.
3. Upload one video for overlay analysis, or two videos for comparison modes.
4. Adjust the pose model, inference resolution, and smoothing options if needed.
5. Review the overlay, contact frame, and metrics output.

## Tips for better results

- Record from the **side view** when possible.
- Use good lighting.
- Keep the climber’s full body in frame.
- Prefer 720p or 1080p video.
- Use 24 FPS or higher.
- Increase inference resolution for higher-quality videos if you want better keypoint precision.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `streamlit: command not found` | Run `python -m streamlit run app.py` |
| `ModuleNotFoundError: No module named 'src'` | Run Streamlit from the project folder (the directory that contains `app.py` and the `src` package). The app adds that folder to Python’s path automatically; if the error persists, use `python -m streamlit run app.py` with your venv activated. |
| Missing module errors | Reinstall dependencies with `pip install -r requirements.txt` |
| Model download fails | Ensure internet access on first run |
| Processing is slow | Use a smaller pose model or standard inference resolution |
| Overlay is jittery | Enable smoothing |
| No pose detected | Make sure the climber is fully visible and well lit |

## Project goal

This project was built to help climbers understand their movement better.
