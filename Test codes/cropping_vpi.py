import cv2
import vpi
import numpy as np
from PIL import Image

# Load the source and template images using OpenCV
image_path = 'Cropping_model/1 (2).png'  # Replace with your source image path
template_path = 'Cropped_data/1.png'  # Replace with your template image path

# Read the images (using OpenCV)
src = cv2.imread(image_path, cv2.IMREAD_COLOR)
templ = cv2.imread(template_path, cv2.IMREAD_COLOR)

if src is None:
    raise ValueError(f"Error loading source image: {image_path}")
if templ is None:
    raise ValueError(f"Error loading template image: {template_path}")

# Convert source and template images to RGB for consistency
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
templ_rgb = cv2.cvtColor(templ, cv2.COLOR_BGR2RGB)

# Convert to VPI format using the backend for CUDA
with vpi.Backend.CUDA:
    # Convert source and template images to VPI image format
    vpi_src = vpi.asimage(src_rgb)
    vpi_templ = vpi.asimage(templ_rgb)

    # Perform template matching
    output = vpi.templateMatching(vpi_src, vpi_templ)
  
    # Convert the output to float32 format for processing
    temp = output.convert(vpi.Format.F32, backend=vpi.Backend.CUDA, scale=255)

    # Find the location of the best match
    min_coords, max_coords = temp.minmaxloc(min_capacity=10000, max_capacity=10000)

    # Convert the result back to 8-bit format for visualization
    output = temp.convert(vpi.Format.U8, backend=vpi.Backend.CUDA)

# Visualize the result (with OpenCV)
# Draw a rectangle around the matched region
top_left = max_coords
bottom_right = (top_left[0] + templ.shape[1], top_left[1] + templ.shape[0])

# Draw the rectangle on the source image
cv2.rectangle(src, top_left, bottom_right, (0, 255, 0), 2)  # Green rectangle for matching region

# Convert to RGB for displaying
src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

# Display the result with OpenCV
cv2.imshow("Template Matching Result", src_rgb)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the result
output_image_path = 'output_template_matching_result.png'
cv2.imwrite(output_image_path, src)  # Save the result with the rectangle
print(f"Result saved as '{output_image_path}'")
