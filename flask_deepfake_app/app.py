import os
import cv2 # OpenCV for video processing
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image # Pillow for image loading/processing
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import time # For unique filenames

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Store uploaded videos temporarily
STATIC_FOLDER = 'static' # Serve extracted frame from here
TEMP_FRAME_FOLDER = os.path.join(STATIC_FOLDER, 'temp_frames') # Subfolder for frames
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'} # Allowed video formats
MODEL_PATH = 'deepfake_detector_resnet18.pth' # Path to your saved model
NUM_CLASSES = 2
CLASS_NAMES = ['fake', 'real'] # MUST match training order (usually alphabetical)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Initialize Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['SECRET_KEY'] = 'your_very_secret_key' # Change for production!
# Ensure upload and static frame directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FRAME_FOLDER, exist_ok=True)

# --- Load Pre-trained Model ---
print(f"Loading model on {DEVICE}...")
# Define the model architecture (must match training)
model = models.resnet18(weights=None) # Load architecture only
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Load the trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}. Place the model file in the correct directory.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load model weights: {e}")
    exit()

# Set model to evaluation mode (important!)
model.eval()
model.to(DEVICE)
print("Model loaded successfully.")

# --- Image Preprocessing Transform ---
# Must match the validation transform used during training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frame_from_video(video_path, output_image_path):
    """Extracts the middle frame from a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Video has no frames: {video_path}")
            cap.release()
            return False

        # Go to the middle frame
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"Failed to read frame {middle_frame_index} from {video_path}")
            return False

        # Save the frame
        cv2.imwrite(output_image_path, frame)
        print(f"Successfully extracted frame to {output_image_path}")
        return True
    except Exception as e:
        print(f"Error extracting frame from {video_path}: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return False

def predict_frame(image_path, model, preprocess_transform, device, class_names):
    """Predicts if an image frame is real or fake."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = class_names[predicted_idx.item()]

        # Get confidence for each class specifically
        fake_confidence = confidence_scores[class_names.index('fake')]
        real_confidence = confidence_scores[class_names.index('real')]

        return predicted_class, fake_confidence, real_confidence

    except FileNotFoundError:
        print(f"Prediction error: Image file not found at {image_path}")
        return None, 0.0, 0.0
    except Exception as e:
        print(f"Error during prediction for {image_path}: {e}")
        return None, 0.0, 0.0


# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles video upload, frame extraction, and prediction."""
    if 'video' not in request.files:
        flash('No video file part found.')
        return redirect(request.url)

    file = request.files['video']
    if file.filename == '':
        flash('No video selected for uploading.')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Use secure_filename and add a timestamp/unique ID to prevent conflicts
        unique_id = str(int(time.time()))
        original_filename = secure_filename(file.filename)
        base_filename, file_ext = os.path.splitext(original_filename)
        video_filename = f"{base_filename}_{unique_id}{file_ext}"
        frame_filename = f"{base_filename}_{unique_id}.jpg" # Save frame as jpg

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        frame_path_full = os.path.join(TEMP_FRAME_FOLDER, frame_filename) # Full path for saving
        # Path relative to static folder for URL generation in template:
        frame_path_relative = os.path.join('temp_frames', frame_filename)


        video_saved = False
        frame_extracted = False
        try:
            # 1. Save uploaded video
            file.save(video_path)
            video_saved = True
            print(f"Video saved to {video_path}")

            # 2. Extract frame
            if not extract_frame_from_video(video_path, frame_path_full):
                 flash('Error extracting frame from video.')
                 # Render index even on error, but pass the error message
                 return render_template('index.html', error='Could not extract a frame from the video.')
            frame_extracted = True

            # 3. Predict using the model
            prediction, conf_fake, conf_real = predict_frame(frame_path_full, model, preprocess, DEVICE, CLASS_NAMES)

            if prediction is None:
                 flash('Error occurred during prediction.')
                 # Render index even on error, but pass the error message
                 return render_template('index.html', error='Prediction failed.')

            # 4. Render results
            return render_template('result.html',
                                   filename=original_filename,
                                   prediction=prediction,
                                   confidence_fake=conf_fake,
                                   confidence_real=conf_real,
                                   extracted_frame_path=frame_filename) # Pass filename for url_for

        except Exception as e:
             flash(f'An unexpected error occurred: {e}')
             print(f"Unexpected error during prediction process: {e}")
             # Redirect back to index on generic error
             return redirect(url_for('index'))

        finally:
            # 5. Cleanup: Remove temporary files
            if video_saved and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Removed temporary video: {video_path}")
                except OSError as e:
                    print(f"Error removing video file {video_path}: {e}")
            # Keep the frame file temporarily so it can be displayed in the result.
            # A more robust solution would involve a background task for cleanup.
            # For simplicity here, we leave the frame. It will be overwritten if
            # another video with a similar name is uploaded at the exact same second.
            # Consider adding cleanup logic (e.g., delete files older than X minutes)
            # if this becomes a long-running application.

            # If frame extraction failed but video was saved, still try to delete video.
            # The frame file might not exist if extraction failed.

    else:
        flash('Invalid file type. Allowed types are: ' + ', '.join(ALLOWED_EXTENSIONS))
        return redirect(request.url)

    # Fallback redirect if something went wrong before rendering result
    return redirect(url_for('index'))


# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0', port=5000) # Host 0.0.0.0 makes it accessible on network