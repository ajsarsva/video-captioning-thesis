import torch
from PIL import Image
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration


# Load model once globally so we don't reload it for every video
processor = None
model = None


def load_blip_model():
    """Load BLIP model and processor. Call once at the start."""
    global processor, model
    
    if model is None:
        print("Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"BLIP model loaded on {device}!")
    
    return processor, model


def frame_to_caption(frame):
    """
    Generate a caption for a single frame using BLIP.
    
    Args:
        frame: numpy array (BGR format from OpenCV)
        
    Returns:
        caption: generated caption string
    """
    global processor, model
    
    if model is None:
        load_blip_model()
    
    device = next(model.parameters()).device
    
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Generate caption
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption


def frames_to_caption(keyframes):
    """
    Generate one combined caption from multiple keyframes.
    Strategy: caption the middle keyframe as the representative frame.
    
    Args:
        keyframes: list of frames (numpy arrays)
        
    Returns:
        caption: single caption string
    """
    # Use the middle keyframe as most representative
    middle_idx = len(keyframes) // 2
    caption = frame_to_caption(keyframes[middle_idx])
    return caption


if __name__ == "__main__":
    print("BLIP Captioner module loaded successfully!")