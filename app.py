"""Climbing Movement Analysis Tool - Streamlit app."""

import json
import os
import sys
import time
from pathlib import Path


def _find_project_root() -> Path:
    """Directory that contains the `src` package (handles odd cwd / __file__ from Streamlit)."""
    def _has_app_src(base: Path) -> bool:
        return (base / "src" / "pose_estimator.py").is_file()

    here = Path(__file__).resolve().parent
    for base in [here, *list(here.parents)[:8]]:
        if _has_app_src(base):
            return base
    cwd = Path.cwd()
    for base in [cwd, *list(cwd.parents)[:8]]:
        if _has_app_src(base):
            return base
    return here


_ROOT = _find_project_root()
_root_str = str(_ROOT)
if _root_str in sys.path:
    sys.path.remove(_root_str)
sys.path.insert(0, _root_str)

# #region agent log
try:
    with open(_ROOT / "debug-fcbc7d.log", "a", encoding="utf-8") as _agent_f:
        _agent_f.write(
            json.dumps(
                {
                    "sessionId": "fcbc7d",
                    "runId": "run1",
                    "hypothesisId": "H1",
                    "location": "app.py:bootstrap",
                    "message": "project_root",
                    "data": {
                        "file": __file__,
                        "parent_of_file": str(Path(__file__).resolve().parent),
                        "cwd": os.getcwd(),
                        "chosen_root": _root_str,
                        "has_pose_estimator": (_ROOT / "src" / "pose_estimator.py").is_file(),
                        "sys_path_head": sys.path[:4],
                    },
                    "timestamp": int(time.time() * 1000),
                }
            )
            + "\n"
        )
except OSError:
    pass
# #endregion

import streamlit as st
import cv2
import numpy as np
import tempfile

from src.pose_estimator import PoseEstimator, process_video
from src.overlay import draw_pose_overlay
from src.contact_detection import detect_contact_frame
from src.geometry import compute_joint_angles, center_of_gravity
from src.pose_smoother import smooth_landmarks, SMOOTHING_PRESETS


st.set_page_config(page_title="Climb Coach AI", page_icon="🧗", layout="wide")
st.title("🧗 Climbing Movement Analysis")

# Sidebar: mode selection
mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Mode 1: Pose Overlay",
        "Mode 2: Side-by-Side Comparison",
        "Mode 3: Overlapped Contact Frames",
    ],
)

# Initialize session state
if "contact_a" not in st.session_state:
    st.session_state.contact_a = None
if "contact_b" not in st.session_state:
    st.session_state.contact_b = None

# Pose model: larger = more accurate overlay, slower (applies to all modes)
pose_model_label = st.sidebar.selectbox(
    "Pose model",
    ["Nano (fast)", "Small", "Medium (recommended)", "Large", "Extra-large"],
    index=2,
    help="Larger models = more accurate overlay, slower processing.",
)
_model_size_map = {"Nano (fast)": "n", "Small": "s", "Medium (recommended)": "m", "Large": "l", "Extra-large": "x"}
MODEL_SIZE = _model_size_map[pose_model_label]

# Inference resolution: higher = finer keypoints, slower
imgsz_label = st.sidebar.selectbox(
    "Inference resolution",
    ["Standard (640)", "High (1280)"],
    index=0,
    help="High for 720p+ videos. ~2-4x slower but finer keypoint precision.",
)
IMGSZ = 1280 if "High" in imgsz_label else 640

# Temporal smoothing: reduces keypoint jitter
smoothing_strength = st.sidebar.selectbox(
    "Smoothing strength",
    [0, 1, 2, 3],
    format_func=lambda x: ["Off", "Light", "Medium", "Strong"][x],
    index=0,
    help="Smoother overlay = less jitter. Light/Medium recommended.",
)

# User context: camera and wall angle (advisory, for future angle correction)
with st.sidebar.expander("Video context", expanded=False):
    camera_angle = st.selectbox(
        "Camera angle",
        ["Side view (recommended)", "Front view", "Diagonal", "Other / unsure"],
        help="Side view works best for pose detection.",
    )
    wall_angle = st.selectbox(
        "Wall angle",
        ["Slab (< 90°)", "Vertical (90°)", "Slight overhang (100–110°)", "Overhang (> 110°)"],
        help="For future angle correction relative to gravity.",
    )
    if camera_angle != "Side view (recommended)":
        st.caption("Tip: Side view works best for pose detection.")


# Video quality thresholds
MIN_HEIGHT_RECOMMENDED = 720
MIN_FPS_RECOMMENDED = 24
MIN_HEIGHT_OPTIMAL = 1080
MIN_FPS_OPTIMAL = 30
MIN_HEIGHT_BLOCK = 480
MIN_FPS_BLOCK = 15


def _get_video_metadata(path: str):
    """Return (width, height, fps) or None if unreadable."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return (w, h, fps) if (w and h) else None


def _video_quality_panel(width: int, height: int, fps: float):
    """Show advisory or error. Returns True if OK to process, False if blocked."""
    if height < MIN_HEIGHT_BLOCK or fps < MIN_FPS_BLOCK:
        st.error(
            "**Video quality too low** for reliable pose detection. "
            f"Your video: {width}x{height} @ {fps:.0f} FPS. "
            "Please record at 720p (1280×720) or higher, 24+ FPS."
        )
        return False
    good_res = height >= MIN_HEIGHT_RECOMMENDED
    good_fps = fps >= MIN_FPS_RECOMMENDED
    optimal_res = height >= MIN_HEIGHT_OPTIMAL
    optimal_fps = fps >= MIN_FPS_OPTIMAL
    if optimal_res and optimal_fps:
        st.info(f"For best pose accuracy: 1080p+, 30 FPS. Your video: {width}×{height} @ {fps:.1f} FPS ✓")
    elif good_res and good_fps:
        st.info(f"For best pose accuracy: 720p+, 24 FPS. Your video: {width}×{height} @ {fps:.1f} FPS ✓")
    else:
        st.warning(
            f"For best pose accuracy: record at 720p or higher, 24+ FPS. "
            f"Your video: {width}×{height} @ {fps:.1f} FPS."
        )
    return True


def _landmarks_for_display(landmarks_list, fps: float):
    """Apply temporal smoothing when strength > 0."""
    preset = SMOOTHING_PRESETS.get(smoothing_strength)
    if preset is None or not landmarks_list:
        return landmarks_list
    return smooth_landmarks(landmarks_list, fps=fps, **preset)


def load_and_process_video(uploaded_file, model_size: str = None, imgsz: int = None):
    """Save uploaded file, process video, return frames and landmarks."""
    if uploaded_file is None:
        return None, None, None, None

    uploaded_file.seek(0)
    data = uploaded_file.read()
    if len(data) == 0:
        return None, None, None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        metadata = _get_video_metadata(path)
        if metadata:
            w, h, f = metadata
            if not _video_quality_panel(w, h, f):
                return None, None, None, None
        size = model_size if model_size is not None else MODEL_SIZE
        infer_sz = imgsz if imgsz is not None else IMGSZ
        frames, landmarks_list, world_landmarks_list, fps = process_video(path, model_size=size, imgsz=infer_sz)
        return frames, landmarks_list, world_landmarks_list, fps
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# --- Mode 1: Pose Overlay ---
if mode == "Mode 1: Pose Overlay":
    st.header("Pose Overlay with Joint Angles & Center of Gravity")
    video_file = st.file_uploader("Upload climbing video", type=["mp4", "mov", "avi"])

    if video_file:
        cache_key = f"mode1_{video_file.name}_{video_file.size}_{MODEL_SIZE}_{IMGSZ}"
        if cache_key not in st.session_state:
            with st.spinner("Processing video..."):
                result = load_and_process_video(video_file, model_size=MODEL_SIZE, imgsz=IMGSZ)
                if result[0] is None:
                    st.error("Could not process video.")
                    st.stop()
                st.session_state[cache_key] = result

        result = st.session_state.get(cache_key)
        if result is None:
            st.stop()
        frames, landmarks_list, _, fps = result
        display_landmarks = _landmarks_for_display(landmarks_list, fps)
        total_frames = len(frames)

        st.sidebar.subheader("Playback")
        frame_idx = st.sidebar.slider(
            "Frame",
            0,
            max(0, total_frames - 1),
            0,
        )

        if frame_idx < total_frames and frames[frame_idx] is not None:
            frame = frames[frame_idx]
            lm = display_landmarks[frame_idx] if frame_idx < len(display_landmarks) else None
            overlay_frame = draw_pose_overlay(frame, lm)
            st.image(overlay_frame, channels="BGR", use_container_width=True)

        st.caption(f"Frame {frame_idx + 1} / {total_frames} | FPS: {fps:.1f}")
        n_detected = sum(1 for lm in landmarks_list if lm is not None)
        if n_detected == 0 and total_frames > 0:
            st.warning(
                "**No pose detected.** The pose model didn't find a person in this video. "
                "Detection works best when the **full body** is visible (head to feet), "
                "shot from the **side or front**, with good lighting. Climbing poses with extreme angles "
                "or cropped limbs often aren't recognized. Try a video where the climber is fully in frame."
            )
        if st.sidebar.button("Clear cache (re-process video)"):
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            st.rerun()


# --- Mode 2: Side-by-Side Comparison ---
elif mode == "Mode 2: Side-by-Side Comparison":
    st.header("Side-by-Side Movement Comparison")
    col1, col2 = st.columns(2)
    with col1:
        video_a = st.file_uploader("Video A (Attempt 1)", type=["mp4", "mov", "avi"], key="a")
    with col2:
        video_b = st.file_uploader("Video B (Attempt 2)", type=["mp4", "mov", "avi"], key="b")

    if video_a and video_b:
        with st.spinner("Processing videos..."):
            result_a = load_and_process_video(video_a, model_size=MODEL_SIZE, imgsz=IMGSZ)
            result_b = load_and_process_video(video_b, model_size=MODEL_SIZE, imgsz=IMGSZ)

        if result_a[0] is None or result_b[0] is None:
            st.error("Could not process one or both videos.")
        else:
            frames_a, lm_a, _, fps_a = result_a
            frames_b, lm_b, _, fps_b = result_b
            disp_a = _landmarks_for_display(lm_a, fps_a)
            disp_b = _landmarks_for_display(lm_b, fps_b)

            # Contact detection (uses raw landmarks)
            vel_threshold = st.sidebar.slider("Velocity threshold (contact)", 0.005, 0.05, 0.015, 0.001)
            contact_a = st.session_state.contact_a or detect_contact_frame(lm_a, velocity_threshold=vel_threshold)
            contact_b = st.session_state.contact_b or detect_contact_frame(lm_b, velocity_threshold=vel_threshold)

            if contact_a is None:
                contact_a = len(frames_a) - 1
            if contact_b is None:
                contact_b = len(frames_b) - 1

            st.sidebar.write("Contact frames (edit to override auto-detect):")
            contact_a = st.sidebar.number_input("Contact frame A", 0, len(frames_a) - 1, contact_a, key="ca2")
            contact_b = st.sidebar.number_input("Contact frame B", 0, len(frames_b) - 1, contact_b, key="cb2")
            if st.sidebar.button("Reset to auto-detect"):
                st.session_state.contact_a = None
                st.session_state.contact_b = None
                st.rerun()

            # Progress slider: 0-100% maps to 0-contact frame in each video
            progress = st.slider("Progress (start → contact)", 0, 100, 0) / 100.0
            idx_a = int(progress * contact_a) if contact_a > 0 else 0
            idx_b = int(progress * contact_b) if contact_b > 0 else 0

            col1, col2 = st.columns(2)
            with col1:
                if idx_a < len(frames_a):
                    fa = draw_pose_overlay(frames_a[idx_a], disp_a[idx_a] if idx_a < len(disp_a) else None)
                    st.image(fa, channels="BGR", use_container_width=True)
                    st.caption(f"Video A: Frame {idx_a + 1} / {contact_a + 1}")
            with col2:
                if idx_b < len(frames_b):
                    fb = draw_pose_overlay(frames_b[idx_b], disp_b[idx_b] if idx_b < len(disp_b) else None)
                    st.image(fb, channels="BGR", use_container_width=True)
                    st.caption(f"Video B: Frame {idx_b + 1} / {contact_b + 1}")


# --- Mode 3: Overlapped Contact Frames ---
elif mode == "Mode 3: Overlapped Contact Frames":
    st.header("Overlapped Contact Frame Comparison")
    col1, col2 = st.columns(2)
    with col1:
        video_a = st.file_uploader("Video A", type=["mp4", "mov", "avi"], key="a3")
    with col2:
        video_b = st.file_uploader("Video B", type=["mp4", "mov", "avi"], key="b3")

    if video_a and video_b:
        with st.spinner("Processing videos..."):
            result_a = load_and_process_video(video_a, model_size=MODEL_SIZE, imgsz=IMGSZ)
            result_b = load_and_process_video(video_b, model_size=MODEL_SIZE, imgsz=IMGSZ)

        if result_a[0] is None or result_b[0] is None:
            st.error("Could not process one or both videos.")
        else:
            frames_a, lm_a, _, fps_a = result_a
            frames_b, lm_b, _, fps_b = result_b
            disp_a = _landmarks_for_display(lm_a, fps_a)
            disp_b = _landmarks_for_display(lm_b, fps_b)

            vel_threshold = st.sidebar.slider("Velocity threshold", 0.005, 0.05, 0.015, 0.001, key="v3")
            contact_a = st.session_state.contact_a or detect_contact_frame(lm_a, velocity_threshold=vel_threshold)
            contact_b = st.session_state.contact_b or detect_contact_frame(lm_b, velocity_threshold=vel_threshold)
            if contact_a is None:
                contact_a = len(frames_a) - 1
            if contact_b is None:
                contact_b = len(frames_b) - 1

            # Manual override for contact frame
            contact_a = st.sidebar.number_input("Contact frame A", 0, len(frames_a) - 1, contact_a, key="ca")
            contact_b = st.sidebar.number_input("Contact frame B", 0, len(frames_b) - 1, contact_b, key="cb")

            frame_a = frames_a[contact_a]
            frame_b = frames_b[contact_b]
            lm_a_contact = disp_a[contact_a] if contact_a < len(disp_a) else None
            lm_b_contact = disp_b[contact_b] if contact_b < len(disp_b) else None

            # Resize frame_b to match frame_a
            h, w = frame_a.shape[:2]
            frame_b_resized = cv2.resize(frame_b, (w, h))

            # Blend 50/50
            blended = cv2.addWeighted(frame_a, 0.5, frame_b_resized, 0.5, 0)

            # Draw both skeletons in different colors
            from src.overlay import draw_skeleton, draw_landmark_points, draw_cog, draw_angles

            if lm_a_contact is not None:
                draw_skeleton(blended, lm_a_contact, (255, 150, 0), 2)  # Blue
                draw_angles(blended, lm_a_contact, (255, 200, 100))
            if lm_b_contact is not None:
                draw_skeleton(blended, lm_b_contact, (0, 200, 255), 2)  # Orange
                draw_angles(blended, lm_b_contact, (100, 200, 255))

            st.image(blended, channels="BGR", use_container_width=True)
            st.caption("Blue = Video A | Orange = Video B")

            # Metrics table
            angles_a = compute_joint_angles(lm_a_contact) if lm_a_contact is not None else {}
            angles_b = compute_joint_angles(lm_b_contact) if lm_b_contact is not None else {}
            joint_names = ["left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_knee", "right_knee", "left_ankle", "right_ankle"]
            rows = []
            for j in joint_names:
                a_val = angles_a.get(j, 0)
                b_val = angles_b.get(j, 0)
                diff = b_val - a_val
                rows.append({"Joint": j.replace("_", " ").title(), "Video A (°)": f"{a_val:.1f}", "Video B (°)": f"{b_val:.1f}", "Diff (°)": f"{diff:+.1f}"})
            st.dataframe(rows, use_container_width=True, hide_index=True)
