import cv2
import numpy as np
import os
import time

# Record start time before executing the code
start_time = time.time()

# Grayscale threshold for white regions
threshold_value = 200  # Adjust the threshold as necessary

# Function to process the image and save output (background elimination step)
def process_image(image_path, output_prefix):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Resize the image to 1920x1080
    input_image_resized = cv2.resize(input_image, (1920, 1080))

    # Apply white threshold to extract white regions (LEDs)
    _, mask_white = cv2.threshold(input_image_resized, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply the mask to retain only the LED regions
    result_image = cv2.bitwise_and(input_image_resized, input_image_resized, mask=mask_white)

    output_image_path = f'results/{output_prefix}_processed.png'
    cv2.imwrite(output_image_path, result_image)
    return output_image_path


# Function to process an image for contour detection (only based on white regions)
def process_image_for_contours(image_path, output_contour_folder):
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Apply white threshold to extract white regions (LEDs)
    _, binary_thresh = cv2.threshold(input_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of the white regions
    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored contours
    cv2.drawContours(contour_image, contours_white, -1, (0, 255, 0), 2)  # Draw green contours

    # Save the contour-mapped image
    contour_output_path = os.path.join(output_contour_folder, os.path.basename(image_path))
    cv2.imwrite(contour_output_path, contour_image)

    # Count the number of contours (white regions)
    white_region_count = len(contours_white)

    return white_region_count, contour_output_path


# Main function to process images from a single folder
def process_images_in_folder(folder_path, contour_output_folder):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if len(image_files) % 2 != 0:
        print("Warning: The number of images in the folder is odd. One image will not have a pair.")
        image_files = image_files[:-1]

    for i in range(0, len(image_files), 2):
        left_file = image_files[i]
        right_file = image_files[i + 1]

        left_image_path = os.path.join(folder_path, left_file)
        right_image_path = os.path.join(folder_path, right_file)

        left_prefix = os.path.splitext(left_file)[0]
        right_prefix = os.path.splitext(right_file)[0]

        print(f"Processing pair: Left - {left_file}, Right - {right_file}...")

        # Process the images
        left_output = process_image(left_image_path, f'{left_prefix}')
        right_output = process_image(right_image_path, f'{right_prefix}')

        # Process contours for both images
        left_white, left_contour_path = process_image_for_contours(left_output, contour_output_folder)
        right_white, right_contour_path = process_image_for_contours(right_output, contour_output_folder)

        # Compare the number of contours (LED regions)
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
        
        # Save the result
        cv2.imwrite(comparison_output_path, combined_image_resized)

        print(f"Result: {label}. Comparison saved at: {comparison_output_path}")
        print(f"Left contour image saved at: {left_contour_path}")
        print(f"Right contour image saved at: {right_contour_path}")

    print("All image pairs processed!")


# Set the folder paths
folder_path = 'NEW_TEST_OWN/new_30fps'
contour_output_folder = 'NEW_TEST_OWN/output_images'

# Create the output directories if they do not exist
os.makedirs('results', exist_ok=True)
os.makedirs(contour_output_folder, exist_ok=True)

# Process the images in the folder
process_images_in_folder(folder_path, contour_output_folder)

# Record end time after execution is complete
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Clean up and close any open windows
cv2.destroyAllWindows()
