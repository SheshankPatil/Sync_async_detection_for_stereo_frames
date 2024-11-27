import cv2
import vpi
import numpy as np
from PIL import Image

# Define two input image paths
image_path1 = 'Test dataset/crooped_template.png'  # Replace with your first image path
image_path2 = 'Test dataset/crooped_template.png'  # Replace with your second image path

# Red color threshold ranges (in HSV space)
lower_bright_red1 = np.array([0, 150, 150])  # Lower bound for red hue near 0°
upper_bright_red1 = np.array([10, 255, 255])  # Upper bound for red hue near 0°

lower_bright_red2 = np.array([170, 150, 150])  # Lower bound for red hue near 180°
upper_bright_red2 = np.array([180, 255, 255])  # Upper bound for red hue near 180°

# Grayscale threshold for white regions
threshold_value = 200  # Threshold to isolate white regions

# Function to process the image and save output
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

# Process both images
process_image(image_path1, 'image1')
process_image(image_path2, 'image2')
