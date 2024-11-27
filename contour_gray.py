
import cv2
import numpy as np
import vpi

# Load the input image
image_path = 'output/result_image.png'  # Replace with your image path
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

      threshold_value = 200 # Threshold to isolate white regions
# Since we don't have access to a grayscale VPI image directly, use OpenCV for thresholding
      input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale using OpenCV

      threshold_value = 200
    # Step 1: Threshold to isolate bright regions
      _, binary_thresh = cv2.threshold(input_image_gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours of the bright regions
      contours_bright, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Create an output image to draw contours
      output_image = input_image.copy()

    # Draw contours on the output image (Green for bright regions)
      cv2.drawContours(output_image, contours_bright, -1, (0, 255, 0), 2)

    # Step 3: Get bounding box coordinates for each contour
      bright_region_coords = [cv2.boundingRect(contour) for contour in contours_bright]
      bright_region_count = len(bright_region_coords)

      print("Count of bright regions: ", bright_region_count)

    # Show the output image using OpenCV (for local environments, you can use cv2.imshow)
      cv2.imshow('Bright regions', output_image)

    # Wait for key press and close the window
      cv2.waitKey(0)
      cv2.destroyAllWindows()