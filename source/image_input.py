import cv2
import numpy as np
import os
import time
from get_image import fetch_image

def get_image_input_from_folder(relative_folder_path, num_frames, processed_file="processed_files.txt", wait_time=5):
    """
    Load images from a folder and return a list of processed frames, excluding previously processed files.
    If the number of available images is insufficient, process as many images as possible.

    :param relative_folder_path: Relative path to the folder containing image files
    :param num_frames: Number of frames to process
    :param processed_file: Relative path to the text file storing processed file names
    :param wait_time: Time to wait (in seconds) before rechecking the folder for new images
    :return: List of processed frames (grayscale)
    """
    # Get absolute paths based on the current script's directory
    base_dir = os.path.dirname(__file__)
    folder_path = os.path.join(base_dir, relative_folder_path)
    processed_file_path = os.path.join(base_dir, processed_file)

    # Load previously processed files
    if os.path.exists(processed_file_path):
        with open(processed_file_path, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        processed_files = set()

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    # Get all image file paths in the folder, excluding already processed files
    image_paths = [os.path.join(folder_path, file) 
                    for file in os.listdir(folder_path) 
                    if file.lower().endswith(supported_formats) and file not in processed_files]

    if len(image_paths) == 0:
        print("No images available in the folder. Waiting for new images...")
        time.sleep(wait_time)  # Wait before checking the folder again
        fetch_image()

    # If fewer images than requested, process available images
    if len(image_paths) < num_frames:
        print(f"Only {len(image_paths)} images available. Processing them.")
        num_frames = len(image_paths)


    frames = []
    processed_this_run = []

    for i in range(num_frames):
        # Read the image
        image_path = image_paths[i % len(image_paths)]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        frames.append(image)
        processed_this_run.append(os.path.basename(image_path))
        cv2.imshow("Image Feed", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting image feed...")
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Update the processed files list
    with open(processed_file_path, "a") as f:
        for file_name in processed_this_run:
            f.write(file_name + "\n")

    return np.array(frames)