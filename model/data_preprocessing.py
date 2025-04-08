import cv2
import os
import argparse
from tqdm import tqdm

def extract_frame(video_path, output_folder, frame_choice='middle', output_format='jpg'):
    """
    Extracts a single frame from a video file and saves it as an image.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where the extracted frame will be saved.
        frame_choice (str or int):
            - 'middle': Extract the middle frame.
            - 'first': Extract the very first frame (index 0).
            - int: Extract the frame at this specific index.
        output_format (str): Image format to save (e.g., 'jpg', 'png').

    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return False

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = cap.get(cv2.CAP_PROP_FPS) # Uncomment if needed

        if total_frames <= 0:
            print(f"Warning: Video file has no frames or is invalid: {video_path}")
            cap.release()
            return False

        # Determine the frame index to extract
        frame_index = -1
        if frame_choice == 'middle':
            frame_index = total_frames // 2
        elif frame_choice == 'first':
            frame_index = 0
        elif isinstance(frame_choice, int):
            if 0 <= frame_choice < total_frames:
                frame_index = frame_choice
            else:
                print(f"Error: Frame index {frame_choice} out of bounds "
                      f"(0-{total_frames - 1}) for video: {video_path}")
                cap.release()
                return False
        else:
             print(f"Error: Invalid frame_choice '{frame_choice}' for video: {video_path}")
             cap.release()
             return False

        # Set the video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # Read the frame
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"Error: Could not read frame {frame_index} from video: {video_path}")
            cap.release()
            return False

        # Construct output filename
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_filename}.{output_format}"
        output_path = os.path.join(output_folder, output_filename)

        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Save the frame
        cv2.imwrite(output_path, frame)
        # print(f"Saved frame {frame_index} from {video_path} to {output_path}") # Verbose output

        # Release the video capture object
        cap.release()
        return True

    except Exception as e:
        print(f"An unexpected error occurred processing {video_path}: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return False

def process_video_dataset(video_root_dir, image_output_dir, frame_choice='middle'):
    """
    Processes all videos in a structured dataset directory, extracting one frame per video.

    Args:
        video_root_dir (str): Path to the root directory of the video dataset
                              (containing subfolders like 'real', 'fake', 'train', 'validation').
        image_output_dir (str): Path to the root directory where the output image dataset
                                will be created.
        frame_choice (str or int): Frame selection strategy passed to extract_frame.
    """
    print(f"Starting frame extraction from: {video_root_dir}")
    print(f"Outputting images to: {image_output_dir}")
    print(f"Frame selection method: {frame_choice}")

    processed_count = 0
    failed_count = 0
    skipped_count = 0
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv') # Add more if needed

    # Walk through the directory structure
    for root, dirs, files in os.walk(video_root_dir):
        # Find relative path from the root video dir to the current dir
        relative_path = os.path.relpath(root, video_root_dir)

        # Construct the corresponding output directory path
        current_output_dir = os.path.join(image_output_dir, relative_path)

        video_files = [f for f in files if f.lower().endswith(video_extensions)]

        if not video_files:
            if relative_path != '.': # Don't warn for the root dir itself if empty
                 # Check if it's just an empty directory or a split directory (like 'train')
                 is_split_dir = any(os.path.isdir(os.path.join(root, d)) for d in dirs)
                 if not is_split_dir and not dirs: # Only warn if it seems like a leaf dir with no videos
                     print(f"Info: No video files found in: {root}")
            continue # Skip directories without videos

        print(f"Processing directory: {root} -> {current_output_dir}")

        # Use tqdm for a progress bar over the files in the current directory
        for filename in tqdm(video_files, desc=f"Processing {relative_path}", unit="video"):
            video_path = os.path.join(root, filename)
            output_image_filename = os.path.splitext(filename)[0] + '.jpg' # Force jpg output
            output_image_path_check = os.path.join(current_output_dir, output_image_filename)

            # Optional: Skip if image already exists
            # if os.path.exists(output_image_path_check):
            #     # print(f"Skipping {video_path}, output image already exists.")
            #     skipped_count += 1
            #     continue

            if extract_frame(video_path, current_output_dir, frame_choice=frame_choice, output_format='jpg'):
                processed_count += 1
            else:
                failed_count += 1

    print("\n--- Extraction Summary ---")
    print(f"Successfully processed videos: {processed_count}")
    print(f"Skipped (already exist):    {skipped_count}") # Only relevant if skipping is enabled
    print(f"Failed extractions:         {failed_count}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a single frame from videos in a dataset structure.")
    parser.add_argument("video_dir", help="Path to the root directory of the input video dataset.")
    parser.add_argument("image_dir", help="Path to the root directory for the output image dataset.")
    parser.add_argument(
        "-f", "--frame",
        default="middle",
        help="Which frame to extract: 'middle', 'first', or an integer frame index (default: middle)."
    )

    args = parser.parse_args()

    # Handle integer frame choice
    frame_choice_arg = args.frame
    if frame_choice_arg.isdigit():
        try:
            frame_choice_arg = int(frame_choice_arg)
            if frame_choice_arg < 0:
                 raise ValueError("Frame index cannot be negative.")
        except ValueError as e:
            print(f"Error: Invalid frame index '{args.frame}'. Must be 'middle', 'first', or a non-negative integer. {e}")
            exit(1)
    elif frame_choice_arg not in ['middle', 'first']:
        print(f"Error: Invalid frame choice '{args.frame}'. Must be 'middle', 'first', or a non-negative integer.")
        exit(1)


    if not os.path.isdir(args.video_dir):
        print(f"Error: Input video directory not found: {args.video_dir}")
        exit(1)

    process_video_dataset(args.video_dir, args.image_dir, frame_choice=frame_choice_arg)