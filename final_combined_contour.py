import cv2
import numpy as np

# Function to process an image and return contour counts
def process_image(image_path):
    # Load the input image
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


# Paths to the two input images
image_path1 = 'output_image1_red_and_white_regions_pil.png'  # Replace with your first image path
image_path2 = 'output_image2_red_and_white_regions_pil.png'  # Replace with your second image path

# Process both images
red_count1, white_count1, output_image1 = process_image(image_path1)
red_count2, white_count2, output_image2 = process_image(image_path2)

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
