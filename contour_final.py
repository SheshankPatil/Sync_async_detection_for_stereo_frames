import cv2
import numpy as np
import vpi

# Load the input image
image_path = 'output_red_regions_pil.png'  # Replace with your image path
input_image = cv2.imread(image_path)
print("Image read")

# Check if the image was loaded correctly
if input_image is None:
    raise ValueError("Error loading image")

# Convert the image from BGR to RGB format
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Step 1: Create a VPI image using the input image dimensions (use shape attributes)
height, width, _ = input_image_rgb.shape  # Extract height, width, and channels
with vpi.Backend.CUDA:
    # Create a VPI image with grayscale format (U8 is 8-bit grayscale)
    output_vpi_image = vpi.Image((width, height), vpi.Format.U8)

    # Step 2: Convert the image to HSV color space
    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

    # Step 3: Define the red color range in HSV (Red spans two regions in hue)
    lower_red1 = np.array([0, 150, 150])  # Lower bound for red hue near 0°
    upper_red1 = np.array([10, 255, 255])  # Upper bound for red hue near 0°

    lower_red2 = np.array([170, 150, 150])  # Lower bound for red hue near 180°
    upper_red2 = np.array([180, 255, 255])  # Upper bound for red hue near 180°

    # Step 4: Create masks for red color ranges
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two red masks
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    # Step 5: Find contours of the red regions
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 6: Create an output image to draw contours
    output_image = input_image.copy()

    # Step 7: Draw contours on the output image (Red for red regions)
    cv2.drawContours(output_image, contours_red, -1, (0, 0, 255), 2)  # Red color for contours

    # Step 8: Get bounding box coordinates for each red contour
    red_region_coords = [cv2.boundingRect(contour) for contour in contours_red]
    red_region_count = len(red_region_coords)

    print("Count of red regions: ", red_region_count)

    # Show the output image with contours of red regions
    cv2.imshow('Red regions', output_image)

    # Wait for key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
