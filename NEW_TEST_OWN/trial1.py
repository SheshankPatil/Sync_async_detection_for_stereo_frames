import cv2
import numpy as np
import os
import time
import re
from PIL import Image

# Record start time before executing the code
start_time = time.time()

threshold_value = 220  # Grayscale threshold for white regions

# Function to process the image and save output (background elimination step)
def process_image(image_path, output_prefix):
    input_image = Image.open(image_path).convert('L')  # Convert image to grayscale
    input_data = np.asarray(input_image, dtype=np.uint8)
    
    # Apply white threshold to create the mask for white regions
    _, mask_white = cv2.threshold(input_data, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Create the result image by masking out non-white regions
    result_image = cv2.bitwise_and(input_data, input_data, mask=mask_white)
    
    # Save the processed image in the 'results1' folder
    output_image_path = f'results1/{output_prefix}_processed.png'
    cv2.imwrite(output_image_path, result_image)
    return output_image_path

# Function to process an image for contour detection
def process_image_for_contours(image_path):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    _, binary_thresh = cv2.threshold(input_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    white_region_count = len(contours_white)
    
    return white_region_count

# Function to extract the timestamp from the filename
def extract_timestamp(filename):
    # Use regex to extract the timestamp (e.g., '20241128_062506_757099' from 'right_20241128_062506_757099.png')
    match = re.search(r'(\d{8}_\d{6}_\d+)', filename)
    return match.group(0) if match else None

# Main function to process images from two folders (left and right)
def process_images_in_folders(left_folder_path, right_folder_path):
    # Read the images from both folders
    left_image_files = [f for f in os.listdir(left_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    right_image_files = [f for f in os.listdir(right_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    
    # Sort the files using the timestamp extracted from filenames
    left_image_files.sort(key=lambda f: extract_timestamp(f))
    right_image_files.sort(key=lambda f: extract_timestamp(f))

    # Pair images with matching timestamps
    paired_files = []
    left_idx = 0
    right_idx = 0
    
    while left_idx < len(left_image_files) and right_idx < len(right_image_files):
        left_timestamp = extract_timestamp(left_image_files[left_idx])
        right_timestamp = extract_timestamp(right_image_files[right_idx])
        
        if left_timestamp == right_timestamp:
            # If the timestamps match, pair them together
            paired_files.append((left_image_files[left_idx], right_image_files[right_idx]))
            left_idx += 1
            right_idx += 1
        elif left_timestamp < right_timestamp:
            # If the left timestamp is earlier, move to the next left image
            left_idx += 1
        else:
            # If the right timestamp is earlier, move to the next right image
            right_idx += 1

    # Process the paired images
    for left_file, right_file in paired_files:
        left_image_path = os.path.join(left_folder_path, left_file)
        right_image_path = os.path.join(right_folder_path, right_file)

        left_prefix = os.path.splitext(left_file)[0]
        right_prefix = os.path.splitext(right_file)[0]

        print(f"Processing pair: Left - {left_file}, Right - {right_file}...")

        # Process the images
        left_output = process_image(left_image_path, f'{left_prefix}')
        right_output = process_image(right_image_path, f'{right_prefix}')

        # Process for contours
        left_white = process_image_for_contours(left_output)
        right_white = process_image_for_contours(right_output)

        total_left = left_white
        total_right = right_white

        # Save comparison results in the 'results1' folder
        comparison_output_path = f'results1/{left_prefix}_{right_prefix}_comparison.png'

        left_image = cv2.imread(left_output)
        right_image = cv2.imread(right_output)

        # Ensure both images are the same size by resizing them to the smaller one
        if left_image.shape != right_image.shape:
            # Resize both images to the size of the smaller one
            height = min(left_image.shape[0], right_image.shape[0])
            width = min(left_image.shape[1], right_image.shape[1])
            left_image = cv2.resize(left_image, (width, height))
            right_image = cv2.resize(right_image, (width, height))

        # Ensure both images are the same type (e.g., uint8)
        if left_image.dtype != right_image.dtype:
            left_image = left_image.astype(np.uint8)
            right_image = right_image.astype(np.uint8)

        # Concatenate images horizontally
        combined_image = cv2.hconcat([left_image, right_image])

        # Resize the combined image to 1920x1080 for consistency
        combined_image_resized = cv2.resize(combined_image, (1920, 1080))

        # Add label for Sync/Async based on contour comparison
        label = "Sync" if total_left == total_right else "Async"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image_resized, label, (10, 50), font, 1, (0, 255, 0) if label == "Sync" else (0, 0, 255), 2)

        # Save the result of the comparison
        cv2.imwrite(comparison_output_path, combined_image_resized)

        print(f"Result: {label}. Comparison saved at: {comparison_output_path}")

    print("All image pairs processed!")


# Call the function with your folder path
left_folder_path = 'NEW_TEST_OWN/new_30fps/small_left'
right_folder_path = 'NEW_TEST_OWN/new_30fps/small_right'
os.makedirs('results1', exist_ok=True)  # Create 'results1' folder if it doesn't exist
process_images_in_folders(left_folder_path, right_folder_path)

# Record end time after execution is complete
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Clean up and close any open windows
cv2.destroyAllWindows()
