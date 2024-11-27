import cv2
import vpi
import numpy as np
from PIL import Image

# Load the input image
image_path = 'output/resized_1.png'  # Replace with your image path


with vpi.Backend.CUDA:


# Example: Loading an input image and converting it to grayscale using PIL
    input_image = Image.open(image_path).convert('L')  # Convert to grayscale

    input = vpi.asimage(np.asarray(Image.open(image_path)))
    print("Image read")

# Convert the image to numpy array
    input_data = np.asarray(input_image, dtype=np.uint8)

# Convert numpy array to vpi.Image (for template matching or other operations)
    input_vpi = vpi.asimage(input_data)

# Convert vpi.Image to numpy array before using OpenCV functions
    input_np = input_vpi.cpu()  # This gets the image as a numpy array (if it was originally a vpi.Image)

# Perform the threshold operation (assuming input_np is a numpy array)
    threshold_value = 200 
    _, mask_white = cv2.threshold(input_np, threshold_value, 255, cv2.THRESH_BINARY)

# Now you can use cv2.bitwise_and with valid numpy arrays
    result_image = cv2.bitwise_and(input_np, input_np, mask=mask_white)

# Save the result
    cv2.imwrite('output1.png', result_image)

# Convert the result back to an image format and save it as a PNG
    Image.fromarray(result_image).save('output2.png')

    print("Result images saved as 'output1.png' and 'output2.png'")
