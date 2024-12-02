import cv2
import numpy as np
from PIL import Image
import os
import time
import re

# Record start time before executing the code
start_time = time.time()

threshold_value = 250  # Grayscale threshold for white regions

# Function to extract a numeric part from the filename for sorting
def extract_numeric_part(filename):
    # Use regular expression to find all numbers in the filename
    numbers = re.findall(r'\d+', filename)
    # If there are any numbers, return the first one (assuming it's the one used for sorting)
    if numbers:
        return int(numbers[0])
    return 0  # Default to 0 if no numbers are found

# Sort the files using the extracted numeric part
def sort_files(files):
    files.sort(key=extract_numeric_part)

# Function to process the image and save output (background elimination step)
def process_image(image_path, output_prefix):
    input_image = Image.open(image_path).convert('RGB')
    input_data = np.asarray(input_image, dtype=np.uint8)

    # Resize the image to 1920x1080
    input_image_resized = cv2.resize(input_data, (1920, 1080))

    input_gray = cv2.cvtColor(input_image_resized, cv2.COLOR_RGB2GRAY)
    _, mask_white = cv2.threshold(input_gray, threshold_value, 255, cv2.THRESH_BINARY)

    result_image = cv2.bitwise_and(input_image_resized, input_image_resized, mask=mask_white)

    output_image_path = f'results/{output_prefix}_processed.png'
    cv2.imwrite(output_image_path, result_image)
    return output_image_path

# Function to process an image for contour detection
def process_image_for_contours(image_path, contour_output_folder):
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save contour-mapped image for white contours only
    white_contour_image = cv2.drawContours(np.zeros_like(input_image), contours_white, -1, (0, 255, 0), 2)

    white_contour_output_path = os.path.join(contour_output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_white_contours.png")

    cv2.imwrite(white_contour_output_path, white_contour_image)

    white_region_count = len(contours_white)

    return white_region_count

# Main function to process images from left and right folders
def process_images_in_folders(left_folder_path, right_folder_path, contour_output_folder):
    # Get all image files in the left and right folders
    left_files = [f for f in os.listdir(left_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    right_files = [f for f in os.listdir(right_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    # Sort the files using the extract_numeric_part method
    sort_files(left_files)
    sort_files(right_files)

    if len(left_files) != len(right_files):
        print("Warning: The number of images in left and right folders do not match.")

    # Process each pair of left and right images
    for left_file, right_file in zip(left_files, right_files):
        left_image_path = os.path.join(left_folder_path, left_file)
        right_image_path = os.path.join(right_folder_path, right_file)

        left_prefix = os.path.splitext(left_file)[0]
        right_prefix = os.path.splitext(right_file)[0]

        print(f"Processing pair: Left - {left_file}, Right - {right_file}...")

        # Process the images
        left_output = process_image(left_image_path, f'{left_prefix}')
        right_output = process_image(right_image_path, f'{right_prefix}')

        # Process contours for both images
        left_white = process_image_for_contours(left_output, contour_output_folder)
        right_white = process_image_for_contours(right_output, contour_output_folder)

        total_left = left_white
        total_right = right_white

        comparison_output_path = f'results/{left_prefix}_{right_prefix}_comparison.png'

        # Read the processed images
        left_image = cv2.imread(left_output)
        right_image = cv2.imread(right_output)

        # Check and resize images if necessary to ensure same dimensions
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

        # Resize the combined image to 1920x1080
        combined_image_resized = cv2.resize(combined_image, (1920, 1080))

        # Add label for Sync/Async based on contour comparison
        label = "Sync" if total_left == total_right else "Async"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image_resized, label, (10, 50), font, 1, (0, 255, 0) if label == "Sync" else (0, 0, 255), 2)

        # Save the result of the comparison
        cv2.imwrite(comparison_output_path, combined_image_resized)

        print(f"Result: {label}. Comparison saved at: {comparison_output_path}")

    print("All image pairs processed!")

# Set paths for left, right folders and the contour output folder
left_folder_path = 'NEW_TEST_OWN/new_30fps/small_left'
right_folder_path = 'NEW_TEST_OWN/new_30fps/small_right'
contour_output_folder = 'NEW_TEST_OWN/output_images'

# Create the results folder if it doesn't exist
os.makedirs('results', exist_ok=True)
os.makedirs(contour_output_folder, exist_ok=True)

# Process the images in the specified folders
process_images_in_folders(left_folder_path, right_folder_path, contour_output_folder)

# Record end time after execution is complete
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Clean up and close any open windows
cv2.destroyAllWindows()
