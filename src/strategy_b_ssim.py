import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_ssim(frame1, frame2):
    """
    Compute SSIM score between two frames.
    Lower score = more different = scene change.
    
    Args:
        frame1, frame2: numpy arrays (BGR format)
        
    Returns:
        score: float between 0 and 1
    """
    # Convert to grayscale for faster computation
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Resize to smaller size for speed
    gray1 = cv2.resize(gray1, (160, 120))
    gray2 = cv2.resize(gray2, (160, 120))
    
    score, _ = ssim(gray1, gray2, full=True)
    return score


def ssim_sampling(frames, threshold=0.7, max_keyframes=8):
    """
    Select keyframes based on SSIM scene change detection.
    When SSIM drops below threshold, a scene change is detected.
    
    Args:
        frames: list of frames from extract_frames()
        threshold: SSIM score below this = scene change (default 0.7)
        max_keyframes: maximum number of keyframes to return (default 8)
        
    Returns:
        keyframes: list of selected frames
        indices: list of selected frame indices
        ssim_scores: list of SSIM scores between consecutive frames
    """
    
    if len(frames) == 0:
        raise ValueError("No frames provided")
    
    if len(frames) == 1:
        return frames, [0], []
    
    # Always include the first frame
    keyframe_indices = [0]
    ssim_scores = []
    
    # Compare consecutive frames
    for i in range(1, len(frames)):
        score = compute_ssim(frames[i-1], frames[i])
        ssim_scores.append(score)
        
        # Scene change detected
        if score < threshold:
            keyframe_indices.append(i)
    
    # If too many keyframes, keep the most significant ones
    # (those with lowest SSIM scores = biggest scene changes)
    if len(keyframe_indices) > max_keyframes:
        # Always keep first frame
        # Sort remaining by their SSIM score (ascending = most different first)
        scored = []
        for idx in keyframe_indices[1:]:
            if idx > 0:
                scored.append((ssim_scores[idx-1], idx))
        
        # Sort by score ascending (biggest changes first)
        scored.sort(key=lambda x: x[0])
        
        # Take top (max_keyframes - 1) most significant changes + first frame
        top_indices = [idx for _, idx in scored[:max_keyframes-1]]
        keyframe_indices = sorted([0] + top_indices)
    
    # If too few keyframes, fall back to uniform sampling
    if len(keyframe_indices) < 2:
        step = len(frames) // max_keyframes
        keyframe_indices = list(range(0, len(frames), max(step, 1)))[:max_keyframes]
    
    keyframes = [frames[i] for i in keyframe_indices]
    
    return keyframes, keyframe_indices, ssim_scores


if __name__ == "__main__":
    print("Strategy B: SSIM Scene Change Detection module loaded successfully!")