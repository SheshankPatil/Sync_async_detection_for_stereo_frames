import cv2
import vpi
import numpy as np
from PIL import Image
import os
import time

# Record start time before executing the code
start_time = time.time()

# Red color threshold ranges (in HSV space)
lower_bright_red1 = np.array([0, 150, 150])
upper_bright_red1 = np.array([10, 255, 255])
lower_bright_red2 = np.array([170, 150, 150])
upper_bright_red2 = np.array([180, 255, 255])
threshold_value = 200  # Grayscale threshold for white regions

# Function to process the image and save output (background elimination step)
def process_image(image_path, output_prefix):
    with vpi.Backend.CUDA:
        input_image = Image.open(image_path).convert('RGB')
        input_data = np.asarray(input_image, dtype=np.uint8)
        input_vpi = vpi.asimage(input_data)
        input_np = input_vpi.cpu()

        hsv_image = cv2.cvtColor(input_np, cv2.COLOR_RGB2HSV)
        mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
        mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)
        red_mask = cv2.bitwise_or(mask_red1, mask_red2)

        input_gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY)
        _, mask_white = cv2.threshold(input_gray, threshold_value, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(red_mask, mask_white)
        result_image = cv2.bitwise_and(input_np, input_np, mask=combined_mask)

        output_image_path = f'results/{output_prefix}_processed.png'
        cv2.imwrite(output_image_path, result_image)
        return output_image_path


# Function to process an image for contour detection
def process_image_for_contours(image_path):
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"Error loading image: {image_path}")

    hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv_image, lower_bright_red1, upper_bright_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_bright_red2, upper_bright_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours_white, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_region_count = len(contours_red)
    white_region_count = len(contours_white)

    return red_region_count, white_region_count


# Main function to process images from a single folder
def process_images_in_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if len(image_files) % 2 != 0:
        print("Warning: The number of images in the folder is odd. One image will not have a pair.")
        image_files = image_files[:-1]

    for i in range(0, len(image_files), 2):
        left_file = image_files[i]
        right_file = image_files[i + 1]

        left_image_path = os.path.join(folder_path, left_file)
        right_image_path = os.path.join(folder_path, right_file)

        left_prefix = os.path.splitext(left_file)[0]
        right_prefix = os.path.splitext(right_file)[0]

        print(f"Processing pair: Left - {left_file}, Right - {right_file}...")

        left_output = process_image(left_image_path, f'{left_prefix}')
        right_output = process_image(right_image_path, f'{right_prefix}')

        left_red, left_white = process_image_for_contours(left_output)
        right_red, right_white = process_image_for_contours(right_output)

        total_left = left_red + left_white
        total_right = right_red + right_white

        comparison_output_path = f'results/{left_prefix}_{right_prefix}_comparison.png'

        left_image = cv2.imread(left_output)
        right_image = cv2.imread(right_output)
        combined_image = cv2.hconcat([left_image, right_image])

        label = "Sync" if total_left == total_right else "Async"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined_image, label, (10, 50), font, 1, (0, 255, 0) if label == "Sync" else (0, 0, 255), 2)
        cv2.imwrite(comparison_output_path, combined_image)

        print(f"Result: {label}. Comparison saved at: {comparison_output_path}")

    print("All image pairs processed!")


# Call the function with your folder path
folder_path = 'Cropped_data'
os.makedirs('results', exist_ok=True)
process_images_in_folder(folder_path)

# Record end time after execution is complete
end_time = time.time()
print(f"Execution Time: {end_time - start_time:.2f} seconds")

# Clean up and close any open windows
cv2.destroyAllWindows()

