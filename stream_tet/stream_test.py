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

        # Step 8: Save the result as a PNG file (using OpenCV)
        output_image_path = f'output/{output_prefix}_red_and_white_regions.png'
        cv2.imwrite(output_image_path, result_image)

        # Show the image with red and white regions in real-time
        cv2.imshow(f'Red and White Regions: {output_prefix}', result_image)
        cv2.waitKey(0)  # Wait indefinitely for the user to press any key to continue
        return output_image_path  # Return the path for further processing


# Function to process an image for contour detection (using background-eliminated images)
def process_image_for_contours(image_path):
    # Load the input image (background eliminated)
    input_image = cv2.imread(image_path)

    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # --- Red Region Detection ---
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Create masks for red color ranges
    mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)

    # Combine the two red masks
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Find contours of the red regions
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Bright/White Region Detection ---
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate bright areas (white regions)
    _, binary_thresh = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw contours
    output_image = input_image.copy()

    # Draw red contours (for red regions) in red color
    cv2.drawContours(output_image, contours_red, -1, (0, 0, 255), 2)  # Red color for contours

    # Draw white contours (for bright regions) in green color
    cv2.drawContours(output_image, contours_white, -1, (0, 255, 0), 2)  # Green color for contours

    # Count red and white regions
    red_region_count = len(contours_red)
    white_region_count = len(contours_white)

    # Show the contours image in real-time
    cv2.imshow(f'Contours Image', output_image)
    cv2.waitKey(0)  # Wait indefinitely for the user to press any key to continue

    # Save the output image
    output_image_path = image_path.replace(".png", "_contours.png")
    cv2.imwrite(output_image_path, output_image)

    return red_region_count, white_region_count, output_image_path


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
        # Ensure filenames match
        left_image_path = os.path.join(left_folder_path, left_file)
        right_image_path = os.path.join(right_folder_path, right_file)

        left_prefix = f'{os.path.splitext(left_file)[0]}'
        right_prefix = f'{os.path.splitext(right_file)[0]}'

        print(f"Processing left: {left_file} and right: {right_file}...")

        # Step 1: Background elimination
        left_output_image_path = process_image(left_image_path, left_prefix)
        right_output_image_path = process_image(right_image_path, right_prefix)

        # Step 2: Contour detection
        left_red_count, left_white_count, left_contours_path = process_image_for_contours(left_output_image_path)
        right_red_count, right_white_count, right_contours_path = process_image_for_contours(right_output_image_path)

        print(f"Left {left_file} - Red regions: {left_red_count}, White regions: {left_white_count}, Total: {left_red_count + left_white_count}")
        print(f"Right {right_file} - Red regions: {right_red_count}, White regions: {right_white_count}, Total: {right_red_count + right_white_count}")
        print(f"Left contours saved at: {left_contours_path}")
        print(f"Right contours saved at: {right_contours_path}")

        # Compare results
        if (left_red_count + left_white_count) == (right_red_count + right_white_count):
            print("Sync")
        else:
            print("Async")

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
