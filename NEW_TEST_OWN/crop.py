import cv2
import numpy as np

# Function to crop an image from the bottom to a specific size (1280x579)
def crop_image_from_bottom(image, width=1280, height=579):
    # Get the image dimensions
    img_height, img_width = image.shape[:2]

    # Calculate the cropping region to start from the bottom
    x_start = (img_width - width) // 2  # Center horizontally
    y_start = img_height - height  # Start from the bottom

    # Crop the image from the bottom
    cropped_image = image[y_start:y_start+height, x_start:x_start+width]
    return cropped_image

# Example usage:
# Load an image
input_image = cv2.imread('NEW_TEST_OWN/new_30fps/small_right/right_20241128_065655_919270.png')  # Replace with the path to your image

# Crop the image from the bottom
cropped_image = crop_image_from_bottom(input_image)

# Save the cropped image
cv2.imwrite('cropped_from_bottom_image.png', cropped_image)
