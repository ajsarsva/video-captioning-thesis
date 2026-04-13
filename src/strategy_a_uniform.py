import numpy as np
from frame_extractor import extract_frames


def uniform_sampling(frames, K=8):
    """
    Select K frames evenly spaced across the video.
    
    Args:
        frames: list of frames from extract_frames()
        K: number of keyframes to select (default 8)
        
    Returns:
        keyframes: list of K selected frames
        indices: list of frame indices that were selected
    """
    
    if len(frames) == 0:
        raise ValueError("No frames provided")
    
    if K >= len(frames):
        # If video has fewer frames than K, return all frames
        return frames, list(range(len(frames)))
    
    # Calculate evenly spaced indices
    indices = [int(i * (len(frames) - 1) / (K - 1)) for i in range(K)]
    
    keyframes = [frames[i] for i in indices]
    
    return keyframes, indices


if __name__ == "__main__":
    print("Strategy A: Uniform Sampling module loaded successfully!")