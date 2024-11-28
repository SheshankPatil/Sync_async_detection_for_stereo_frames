import cv2
import vpi
import numpy as np
from PIL import Image
import os
import time  # Import time module

# Record start time before executing the code
start_time = time.time()

# Red color threshold ranges (in HSV space)
lower_bright_red1 = np.array([0, 150, 150])  # Lower bound for red hue near 0°
upper_bright_red1 = np.array([10, 255, 255])  # Upper bound for red hue near 0°

lower_bright_red2 = np.array([170, 150, 150])  # Lower bound for red hue near 180°
upper_bright_red2 = np.array([180, 255, 255])  # Upper bound for red hue near 180°

# Grayscale threshold for white regions
threshold_value = 200  # Threshold to isolate white regions

# Ensure the results folder exists
results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)

# Function to process the image and save output (background elimination step)
def process_image(image_path, output_prefix):
    with vpi.Backend.CUDA:
        # Load and convert the image to RGB using PIL
        input_image = Image.open(image_path).convert('RGB')  # Convert to RGB first

        # Convert the image to a NumPy array (RGB format)
        input_data = np.asarray(input_image, dtype=np.uint8)

        # Convert the image to VPI image format
        input_vpi = vpi.asimage(input_data)

        # Convert the VPI image to a NumPy array for further processing (OpenCV operations)
        input_np = input_vpi.cpu()

        # Step 1: Convert to HSV color space using OpenCV for red region extraction
        hsv_image = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)

        # Step 2: Create masks for red color ranges
        mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)

        # Step 3: Combine both red masks
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        # Step 4: Convert to grayscale for white region extraction
        input_gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY)

        # Step 5: Threshold the grayscale image to isolate white regions
        _, mask_white = cv2.threshold(input_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Step 6: Combine the red and white masks
        combined_mask = cv2.bitwise_or(red_mask, mask_white)

        # Step 7: Apply the combined mask to the input image using bitwise_and
        result_image = cv2.bitwise_and(input_np, input_np, mask=combined_mask)

        return result_image


# Function to process an image for contour detection (using background-eliminated images)
def process_image_for_contours(image):
    # --- Red Region Detection ---
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for red color ranges
    mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)

    # Combine the two red masks
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Find contours of the red regions
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Bright/White Region Detection ---
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate bright areas (white regions)
    _, binary_thresh = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    output_image = image.copy()
    cv2.drawContours(output_image, contours_red, -1, (0, 0, 255), 2)  # Red contours
    cv2.drawContours(output_image, contours_white, -1, (0, 255, 0), 2)  # White contours

    # Count red and white regions
    red_region_count = len(contours_red)
    white_region_count = len(contours_white)

    return red_region_count, white_region_count, output_image


# Main function to process two folders (left and right)
def process_folders(left_folder_path, right_folder_path):
    left_image_files = [f for f in os.listdir(left_folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    right_image_files = [f for f in os.listdir(right_folder_path) if f.endswith('.png') or f.endswith('.jpg')]

    # Sort images by filename (ignoring the extension)
    left_image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    right_image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Ensure both folders have the same number of images
    if len(left_image_files) != len(right_image_files):
        print("Warning: Left and Right folders have different numbers of images!")
        return

    for i, (left_file, right_file) in enumerate(zip(left_image_files, right_image_files)):
        left_image_path = os.path.join(left_folder_path, left_file)
        right_image_path = os.path.join(right_folder_path, right_file)

        print(f"Processing left: {left_file} and right: {right_file}...")

        # Background elimination
        left_result = process_image(left_image_path, f'left_{i}')
        right_result = process_image(right_image_path, f'right_{i}')

        # Contour detection
        left_red_count, left_white_count, left_contours = process_image_for_contours(left_result)
        right_red_count, right_white_count, right_contours = process_image_for_contours(right_result)

        # Combine images for comparison
        combined_result = np.hstack((left_contours, right_contours))

        # Add text to indicate sync or async
        comparison_status = "Sync" if (left_red_count + left_white_count) == (right_red_count + right_white_count) else "Async"
        cv2.putText(combined_result, f"Status: {comparison_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save combined image
        output_image_path = os.path.join(results_folder, f'comparison_{i}_{comparison_status}.png')
        cv2.imwrite(output_image_path, combined_result)

        print(f"Comparison result saved at: {output_image_path}")

    print("All images processed!")

# Call the function with your folder paths (left and right)
left_folder_path = 'Cropped_data/left'  # Replace with the left folder path
right_folder_path = 'Cropped_data/right'  # Replace with the right folder path
process_folders(left_folder_path, right_folder_path)

# Record end time after execution is complete
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Clean up and close any open windows
cv2.destroyAllWindows()

