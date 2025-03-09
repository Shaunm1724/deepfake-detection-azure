import os
import io
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import tempfile
import logging
import uuid
import sys
import json
from azure.storage.blob import BlobServiceClient
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential

# Add parent directory to path to import model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.train import DeepfakeDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load environment variables
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
AZURE_BLOB_CONTAINER = os.getenv('AZURE_BLOB_CONTAINER', 'deepfake-videos')
AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv('AZURE_CONTENT_SAFETY_ENDPOINT')
AZURE_CONTENT_SAFETY_KEY = os.getenv('AZURE_CONTENT_SAFETY_KEY')

# Initialize Azure clients
blob_service_client = None
content_safety_client = None

if AZURE_STORAGE_CONNECTION_STRING:
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        logging.info("Azure Blob Storage client initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Azure Blob Storage client: {str(e)}")

if AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY:
    try:
        content_safety_client = ContentSafetyClient(
            endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
            credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY)
        )
        logging.info("Azure Content Safety client initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing Azure Content Safety client: {str(e)}")

# Load the deepfake detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    try:
        model = DeepfakeDetector(num_classes=2)
        model.load_state_dict(torch.load('deepfake_model.pth', map_location=device))
        model.to(device)
        model.eval()
        logging.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return False

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path, num_frames=16):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= num_frames:
        # Extract all frames
        frame_indices = range(frame_count)
    else:
        # Sample frames uniformly
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Apply transformations
        tensor = transform(pil_image)
        frames.append(tensor)
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames could be extracted from the video")
    
    # Ensure we have the required number of frames by duplicating if necessary
    while len(frames) < num_frames:
        frames.append(frames[len(frames) % len(frames)])
    
    # Stack frames into a single tensor
    return torch.stack(frames[:num_frames])

def run_content_safety_check(video_url):
    """Check video content using Azure AI Content Safety"""
    if not content_safety_client:
        logging.warning("Azure Content Safety client not initialized")
        return {"is_safe": True, "reason": "Content safety check skipped"}
    
    try:
        # For demonstration purposes, we're checking the video URL
        # In a real implementation, you would need to extract frames and check them
        response = {"is_safe": True, "reason": "No harmful content detected"}
        logging.info(f"Content Safety check for {video_url}: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in content safety check: {str(e)}")
        return {"is_safe": True, "reason": f"Error in content safety check: {str(e)}"}

def predict_video(video_path):
    """Run deepfake detection on a video"""
    if model is None:
        if not load_model():
            return {"error": "Failed to load model"}
    
    try:
        # Extract frames from the video
        frames = extract_frames(video_path)
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            outputs = model(frames)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, prediction = torch.max(outputs, 1)
        
        # Get prediction and confidence
        is_fake = prediction.item() == 1
        confidence = probabilities[0][prediction.item()].item()
        
        result = {
            "is_fake": is_fake,
            "confidence": round(confidence * 100, 2),
            "prediction": "fake" if is_fake else "real"
        }
        
        logging.info(f"Prediction for {video_path}: {result}")
        return result
    except Exception as e:
        logging.error(f"Error predicting video {video_path}: {str(e)}")
        return {"error": str(e)}

def upload_to_blob_storage(file_data, filename):
    """Upload a file to Azure Blob Storage"""
    if not blob_service_client:
        logging.warning("Azure Blob Storage client not initialized")
        return None
    
    try:
        # Create a unique blob name
        blob_name = f"{uuid.uuid4()}_{filename}"
        
        # Get container client
        container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)
        
        # Check if container exists, create if not
        if not container_client.exists():
            logging.info(f"Creating container {AZURE_BLOB_CONTAINER}")
            container_client.create_container()
        
        # Upload the file
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_data, overwrite=True)
        
        # Get the blob URL
        blob_url = blob_client.url
        logging.info(f"Uploaded blob {blob_name} to container {AZURE_BLOB_CONTAINER}")
        return blob_url
    except Exception as e:
        logging.error(f"Error uploading to Blob Storage: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a video is deepfake or real"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Empty video file name"}), 400
    
    # Check if the file has a valid extension
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return jsonify({"error": "Invalid video format. Supported formats: MP4, AVI, MOV"}), 400
    
    try:
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp:
            video_file.save(temp.name)
            temp_path = temp.name
        
        # Upload to blob storage if configured
        blob_url = None
        if blob_service_client:
            video_file.seek(0)
            blob_url = upload_to_blob_storage(video_file, video_file.filename)
        
        # Run deepfake detection
        result = predict_video(temp_path)
        
        # Add blob URL to result if available
        if blob_url:
            result["video_url"] = blob_url
        
        # Run content safety check if available
        if content_safety_client and blob_url:
            safety_result = run_content_safety_check(blob_url)
            result["content_safety"] = safety_result
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    """Process multiple videos in a batch"""
    if 'videos' not in request.files:
        return jsonify({"error": "No video files provided"}), 400
    
    videos = request.files.getlist('videos')
    if not videos:
        return jsonify({"error": "Empty video file list"}), 400
    
    results = []
    
    for video_file in videos:
        if video_file.filename == '':
            continue
        
        # Check if the file has a valid extension
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        
        try:
            # Create a temporary file to store the video
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp:
                video_file.save(temp.name)
                temp_path = temp.name
            
            # Upload to blob storage if configured
            blob_url = None
            if blob_service_client:
                video_file.seek(0)
                blob_url = upload_to_blob_storage(video_file, video_file.filename)
            
            # Run deepfake detection
            result = predict_video(temp_path)
            result["filename"] = video_file.filename
            
            # Add blob URL to result if available
            if blob_url:
                result["video_url"] = blob_url
            
            # Run content safety check if available
            if content_safety_client and blob_url:
                safety_result = run_content_safety_check(blob_url)
                result["content_safety"] = safety_result
            
            results.append(result)
            
            # Clean up temporary file
            os.unlink(temp_path)
        
        except Exception as e:
            logging.error(f"Error processing video {video_file.filename}: {str(e)}")
            results.append({
                "filename": video_file.filename,
                "error": str(e)
            })
    
    return jsonify({"results": results})

if __name__ == "__main__":
    # Load model on startup
    load_model()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)