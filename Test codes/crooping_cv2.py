import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the paths for the main image and the template image
main_image_path = 'final/cropping_model/4.png'  # Change this to your local main image path
template_image_path = 'final/cropped_data/1.png'  # Change this to your local template image path

# Load the main image
main_image = cv2.imread(main_image_path)

# Load the template image
template = cv2.imread(template_image_path)

# Check if the images are loaded successfully
if main_image is None or template is None:
    print("Error: Could not load the images.")
    exit()

# Convert both images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Define a threshold to consider a match
threshold = 0.87

# Find locations where the result is above the threshold
locations = np.where(result >= threshold)

# If there are no matches, print "False" and exit
if len(locations[0]) == 0:
    print("False")
    exit()

# Loop over all the locations and mark the matches
for loc in zip(*locations[::-1]):
    cv2.rectangle(main_image, loc, (loc[0] + template.shape[1], loc[1] + template.shape[0]), (0, 255, 0), 2)

# Show the result using matplotlib
plt.imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.axis('off')
plt.show()

# Extract the first matched region
top_left = (locations[1][0], locations[0][0])
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
matched_region = main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Resize the matched region to the specified dimensions (706x463)
output_width, output_height = 1920, 1080
resized_matched_region = cv2.resize(matched_region, (output_width, output_height))

# Ensure the output directory exists before saving
output_dir = 'output'  # Change this to your local output directory path
os.makedirs(output_dir, exist_ok=True)

# Save the resized matched region to a file in the output directory
output_path = os.path.join(output_dir, 'matched_region.png')
cv2.imwrite(output_path, resized_matched_region)

# Print "True"
print("True")

# Display the resized cropped image
plt.imshow(cv2.cvtColor(resized_matched_region, cv2.COLOR_BGR2RGB))
plt.title('Matched Region')
plt.axis('off')
plt.show()
