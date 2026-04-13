import numpy as np
import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans


# Global model variables
clip_model = None
clip_processor = None


def load_clip_model():
    """Load CLIP model. Call once at the start."""
    global clip_model, clip_processor
    
    if clip_model is None:
        print("Loading CLIP model...")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device)
        print(f"CLIP model loaded on {device}!")
    
    return clip_processor, clip_model


def get_frame_embeddings(frames, batch_size=32):
    """
    Get CLIP embeddings for all frames.
    Processes in batches for efficiency.
    
    Args:
        frames: list of frames (numpy arrays BGR)
        batch_size: number of frames to process at once
        
    Returns:
        embeddings: numpy array of shape (num_frames, embedding_dim)
    """
    global clip_model, clip_processor
    
    if clip_model is None:
        load_clip_model()
    
    device = next(clip_model.parameters()).device
    all_embeddings = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        # Convert BGR to RGB PIL images
        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) 
            for f in batch
        ]
        
        # Get CLIP embeddings
        inputs = clip_processor(
            images=pil_images, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            embeddings = clip_model.get_image_features(**inputs)
        
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())
    
    return np.vstack(all_embeddings)


def clip_kmeans_sampling(frames, K=8):
    """
    Select K maximally diverse keyframes using CLIP embeddings + K-Means.
    
    Logic:
    1. Get CLIP embedding for every frame
    2. Cluster embeddings into K groups using K-Means
    3. Pick the frame closest to each cluster center
    
    This ensures selected frames are visually diverse —
    no two selected frames will be redundant.
    
    Args:
        frames: list of frames from extract_frames()
        K: number of keyframes to select (default 8)
        
    Returns:
        keyframes: list of K selected frames
        indices: list of selected frame indices
    """
    
    if len(frames) == 0:
        raise ValueError("No frames provided")
    
    if K >= len(frames):
        return frames, list(range(len(frames)))
    
    # Step 1: Get embeddings for all frames
    embeddings = get_frame_embeddings(frames)
    
    # Step 2: K-Means clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    # Step 3: For each cluster, find the frame closest to cluster center
    indices = []
    for cluster_idx in range(K):
        # Get all frames in this cluster
        cluster_mask = kmeans.labels_ == cluster_idx
        cluster_frame_indices = np.where(cluster_mask)[0]
        
        if len(cluster_frame_indices) == 0:
            continue
        
        # Find frame closest to cluster center
        center = kmeans.cluster_centers_[cluster_idx]
        cluster_embeddings = embeddings[cluster_frame_indices]
        
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_idx = cluster_frame_indices[np.argmin(distances)]
        indices.append(int(closest_idx))
    
    # Sort indices chronologically
    indices = sorted(indices)
    keyframes = [frames[i] for i in indices]
    
    return keyframes, indices


if __name__ == "__main__":
    print("Strategy C: CLIP K-Means module loaded successfully!")