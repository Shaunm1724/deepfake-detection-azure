import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoFrameDataset(Dataset):
    """
    Dataset class for loading frames from videos for deepfake detection.
    
    Args:
        root_dir (str): Directory containing the videos
        num_frames (int): Number of frames to sample from each video
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        
        # Find all video files
        self.video_paths = []
        self.labels = []
        
        # Look for real and fake videos
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        # Check if the directories exist
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise FileNotFoundError(f"Expected directories 'real' and 'fake' under {root_dir}")
        
        # Get real videos (label 0)
        for video_file in os.listdir(real_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                self.video_paths.append(os.path.join(real_dir, video_file))
                self.labels.append(0)  # Real videos are labeled as 0
        
        # Get fake videos (label 1)
        for video_file in os.listdir(fake_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                self.video_paths.append(os.path.join(fake_dir, video_file))
                self.labels.append(1)  # Fake videos are labeled as 1
        
        logging.info(f"Found {len(self.video_paths)} videos ({self.labels.count(0)} real, {self.labels.count(1)} fake)")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get frames from a video and its label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from the video
        frames = self.extract_frames(video_path)
        
        # Convert to torch tensor
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, torch.tensor(label, dtype=torch.long)
    
    def extract_frames(self, video_path):
        """
        Extract frames from a video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # If video has fewer frames than required, we'll duplicate some frames
        if frame_count <= self.num_frames:
            # Read all available frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
            
            # Duplicate frames to reach required count
            while len(frames) < self.num_frames:
                frames.append(frames[len(frames) % len(frames)])
        else:
            # Sample frames uniformly from the video
            frames_to_sample = sorted(random.sample(range(frame_count), self.num_frames))
            frames = []
            
            for frame_idx in frames_to_sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
        
        cap.release()
        return frames

def preprocess_video_dataset(input_dir, output_dir, num_frames=16):
    """
    Preprocess a video dataset by extracting frames from videos.
    
    Args:
        input_dir (str): Directory containing 'real' and 'fake' subdirectories with videos
        output_dir (str): Output directory to save extracted frames
        num_frames (int): Number of frames to extract from each video
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fake'), exist_ok=True)
    
    # Process real videos
    real_dir = os.path.join(input_dir, 'real')
    for video_file in tqdm(os.listdir(real_dir), desc="Processing real videos"):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(real_dir, video_file)
            output_path = os.path.join(output_dir, 'real', os.path.splitext(video_file)[0])
            os.makedirs(output_path, exist_ok=True)
            
            extract_and_save_frames(video_path, output_path, num_frames)
    
    # Process fake videos
    fake_dir = os.path.join(input_dir, 'fake')
    for video_file in tqdm(os.listdir(fake_dir), desc="Processing fake videos"):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(fake_dir, video_file)
            output_path = os.path.join(output_dir, 'fake', os.path.splitext(video_file)[0])
            os.makedirs(output_path, exist_ok=True)
            
            extract_and_save_frames(video_path, output_path, num_frames)

def extract_and_save_frames(video_path, output_path, num_frames):
    """
    Extract frames from a video and save them
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= num_frames:
        # Extract all frames
        frame_indices = range(frame_count)
    else:
        # Sample frames uniformly
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to read frame {frame_idx} from {video_path}")
            continue
        
        # Save the frame
        frame_path = os.path.join(output_path, f'frame_{i:03d}.jpg')
        cv2.imwrite(frame_path, frame)
    
    cap.release()

def prepare_dataset_for_azure(dataset_dir, output_blob_container):
    """
    Prepare dataset metadata for uploading to Azure Blob Storage
    
    Args:
        dataset_dir (str): Directory containing preprocessed dataset
        output_blob_container (str): Name of the Azure Blob Storage container
    
    Returns:
        dict: Dictionary with dataset metadata
    """
    metadata = {
        'real_videos': [],
        'fake_videos': []
    }
    
    # Process real videos
    real_dir = os.path.join(dataset_dir, 'real')
    for video_dir in os.listdir(real_dir):
        video_path = os.path.join(real_dir, video_dir)
        if os.path.isdir(video_path):
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            blob_path = f"real/{video_dir}/"
            metadata['real_videos'].append({
                'video_name': video_dir,
                'frame_count': len(frames),
                'blob_path': blob_path
            })
    
    # Process fake videos
    fake_dir = os.path.join(dataset_dir, 'fake')
    for video_dir in os.listdir(fake_dir):
        video_path = os.path.join(fake_dir, video_dir)
        if os.path.isdir(video_path):
            frames = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
            blob_path = f"fake/{video_dir}/"
            metadata['fake_videos'].append({
                'video_name': video_dir,
                'frame_count': len(frames),
                'blob_path': blob_path
            })
    
    return metadata

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess video dataset for deepfake detection")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory with real and fake videos")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for extracted frames")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to extract per video")
    
    args = parser.parse_args()
    
    preprocess_video_dataset(args.input_dir, args.output_dir, args.num_frames)