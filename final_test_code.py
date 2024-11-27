import cv2
import vpi
import numpy as np
from PIL import Image
import time  # Import time module

# Record start time before executing the code
start_time = time.time()


# Define two input image paths
image_path1 = 'output/image_1_matched_region.png'  # Replace with your first image path
image_path2 = 'output/image_1_matched_region.png'  # Replace with your second image path

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
        output_image_path = f'output_{output_prefix}_red_and_white_regions.png'
        cv2.imwrite(output_image_path, result_image)

        # Convert the result back to PIL Image and save as PNG
        pil_result_image = Image.fromarray(result_image)
        pil_output_path = f'output_{output_prefix}_red_and_white_regions_pil.png'
        pil_result_image.save(pil_output_path)

        print(f"Result image for {output_prefix} saved as '{output_image_path}' and '{pil_output_path}'")

        return pil_output_path  # Return the path for further processing

# Process both images and get output image paths for contour mapping
output_image_path1 = process_image(image_path1, 'image1')
output_image_path2 = process_image(image_path2, 'image2')

# Function to process an image for contour detection (using background-eliminated images)
def process_image_for_contours(image_path):
    # Load the input image (background eliminated)
    input_image = cv2.imread(image_path)

    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    # Convert the image from BGR to RGB format for consistency
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # --- Red Region Detection ---
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 150])  # Lower bound for red hue near 0°
    upper_red1 = np.array([10, 255, 255])  # Upper bound for red hue near 0°

    lower_red2 = np.array([170, 150, 150])  # Lower bound for red hue near 180°
    upper_red2 = np.array([180, 255, 255])  # Upper bound for red hue near 180°

    # Create masks for red color ranges
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two red masks
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Find contours of the red regions
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Bright/White Region Detection ---
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate bright areas (white regions)
    threshold_value = 200  # Set threshold for bright areas
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
    total_count = red_region_count + white_region_count

    # Return the counts and the output image
    return red_region_count, white_region_count, output_image


# Process both output images for contours (background-eliminated)
red_count1, white_count1, output_image1 = process_image_for_contours(output_image_path1)
red_count2, white_count2, output_image2 = process_image_for_contours(output_image_path2)

# Print counts for both images
print(f"Image 1 - Red regions: {red_count1}, White regions: {white_count1}, Total: {red_count1 + white_count1}")
print(f"Image 2 - Red regions: {red_count2}, White regions: {white_count2}, Total: {red_count2 + white_count2}")

# Compare the counts
if (red_count1 + white_count1) == (red_count2 + white_count2):
    print("Sync")
else:
    print("Async")

# Show both images side-by-side for comparison (optional)
combined_output = np.hstack((output_image1, output_image2))  # Horizontal concatenation
cv2.imshow('Comparison of Red and White Contours', combined_output)

# Wait for key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the results to disk
cv2.imwrite('output_with_red_and_white_contours_image1.png', output_image1)
cv2.imwrite('output_with_red_and_white_contours_image2.png', output_image2)


# Record end time after execution is complete
end_time = time.time()

