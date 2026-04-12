import cv2
import os
import numpy as np


def extract_frames(video_path):
    """
    Extract all frames from a video file.
    
    Args:
        video_path: full path to the .mp4 video file
        
    Returns:
        frames: list of frames as numpy arrays (BGR format)
        fps: frames per second of the video
        total_frames: total number of frames in the video
    """
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    return frames, fps, total_frames


def get_video_info(video_path):
    """
    Get basic info about a video without extracting all frames.
    Useful for quick checks.
    """
    cap = cv2.VideoCapture(video_path)
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


if __name__ == "__main__":
    # Quick test — change this path when testing locally
    test_video = "test.mp4"
    info = get_video_info(test_video)
    print("Video info:", info)