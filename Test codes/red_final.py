import cv2
import vpi
import numpy as np
from PIL import Image

# Load the input image
image_path = 'output/resized_1.png'  # Replace with your image path

# Red color threshold ranges (in HSV space)
lower_bright_red1 = np.array([0, 150, 150])  # Lower bound for red hue near 0°
upper_bright_red1 = np.array([10, 255, 255])  # Upper bound for red hue near 0°

lower_bright_red2 = np.array([170, 150, 150])  # Lower bound for red hue near 180°
upper_bright_red2 = np.array([180, 255, 255])  # Upper bound for red hue near 180°

with vpi.Backend.CUDA:
    # Load and convert the image to RGB using PIL
    input_image = Image.open(image_path).convert('RGB')  # Convert to RGB first

    # Convert the image to a NumPy array (RGB format)
    input_data = np.asarray(input_image, dtype=np.uint8)

    # Convert the image to VPI image format
    input_vpi = vpi.asimage(input_data)

    # Convert the VPI image to a NumPy array for further processing (OpenCV operations)
    input_np = input_vpi.cpu()

    # Step 1: Convert to HSV color space using OpenCV
    hsv_image = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)

    # Step 2: Create masks for red color ranges
    mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)

    # Step 3: Combine both red masks
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Step 4: Apply the red mask to the image using bitwise_and
    result_image = cv2.bitwise_and(input_np, input_np, mask=red_mask)

    # Step 5: Save the result as a PNG file
    cv2.imwrite('output_red_regions.png', result_image)

    # Convert the result back to PIL Image and save as PNG
    Image.fromarray(result_image).save('output_red_regions_pil.png')

    print("Result images saved as 'output_red_regions.png' and 'output_red_regions_pil.png'")
