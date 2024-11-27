import cv2
import vpi
import numpy as np

# Load the input image
image_path = 'Test dataset/Left_Camera/captured_0_frame_0012.png' # Replace with your image path
input_image = cv2.imread(image_path)
print("ok")
print("Image read")

# Check if the image was loaded correctly
if input_image is None:
  raise ValueError("Error loading image")

# Convert the image from BGR to RGB format
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Step 1: Create a VPI image using the input image dimensions (use shape attributes)
height, width, _ = input_image_rgb.shape # Extract height, width, and channels

with vpi.Backend.CUDA:
  # Create a VPI image with grayscale format (U8 is 8-bit grayscale)
  output_vpi_image = vpi.Image((width, height), vpi.Format.U8)

# Step 2: Convert the VPI image to grayscale manually if needed (since VPI doesn't directly support color conversion)

# Step 3: Threshold to isolate white regions in the grayscale image
threshold_value = 250 # Threshold to isolate white regions
# Since we don't have access to a grayscale VPI image directly, use OpenCV for thresholding
input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) # Convert to grayscale using OpenCV

# Apply the threshold to the grayscale image
_, mask_white = cv2.threshold(input_image_gray, threshold_value, 255, cv2.THRESH_BINARY)

# Step 4: Mask the input image to retain only the white regions
result_image = cv2.bitwise_and(input_image, input_image, mask=mask_white)

# Display the input and the result images using OpenCV
cv2.imshow('Input Image', input_image)
cv2.imshow('Background Eliminated Image (White Regions)', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
