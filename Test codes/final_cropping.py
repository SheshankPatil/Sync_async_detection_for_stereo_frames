import cv2
import numpy as np
import matplotlib.pyplot as plt
import vpi

# Load the main image
main_image_path = 'final/cropped_data/1.png'
template_image_path = 'final/cropped_data/1.png'

# Load the images using OpenCV
main_image = cv2.imread(main_image_path)
template = cv2.imread(template_image_path)

# Check if the images are loaded successfully
if main_image is None or template is None:
    print("Error: Could not load the images.")
    exit()

# Step 1: Convert to VPI image format
with vpi.Backend.CUDA:  # Use CUDA backend for GPU acceleration
    # Convert the main image and template to VPI format
    vpi_main_image = vpi.asimage(main_image)  # Convert to VPI image
    vpi_template_image = vpi.asimage(template)  # Convert template to VPI image
    
    # Step 2: Create empty grayscale images to hold the converted grayscale versions
    main_gray_vpi = vpi.Image(main_image.shape[:2], vpi.Format.U8)  # Grayscale format (U8)
    template_gray_vpi = vpi.Image(template.shape[:2], vpi.Format.U8)  # Grayscale format (U8)

    # Convert the main and template images to grayscale (in-place)
    #vpi_main_image.convert(vpi.Format.U8, main_gray_vpi)  # Convert to grayscale and store in main_gray_vpi
    binary_image1 = main_gray_vpi.convert(vpi.Format.U8)
    #vpi_template_image.convert(vpi.Format.U8, template_gray_vpi) 
    binary_image2 = template_gray_vpi.convert(vpi.Format.U8)  
     # Convert to grayscale and store in template_gray_vpi

    # Download grayscale images to CPU memory
    main_gray = main_gray_vpi.cpu()
    template_gray = template_gray_vpi.cpu()

# Step 3: Perform template matching using OpenCV (as VPI does not have template matching)
result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# Define a threshold to consider a match
threshold = 0.87

# Step 4: Find locations where the result is above the threshold
locations = np.where(result >= threshold)

# If no matches are found, print "False" and exit
if len(locations[0]) == 0:
    print("False")
    exit()

# Step 5: Loop over all the locations and mark the matches
for loc in zip(*locations[::-1]):
    cv2.rectangle(main_image, loc, (loc[0] + template.shape[1], loc[1] + template.shape[0]), (0, 255, 0), 2)

# Step 6: Show the result using matplotlib
plt.imshow(cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.axis('off')
plt.show()

# Step 7: Extract the first matched region
top_left = (locations[1][0], locations[0][0])
bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
matched_region = main_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Step 8: Resize the matched region to the specified dimensions (1920x1080)
output_width, output_height = 1920, 1080
resized_matched_region = cv2.resize(matched_region, (output_width, output_height))

# Save the resized matched region to a file
output_path = 'output'
cv2.imwrite(output_path, resized_matched_region)

# Print "True"
print("True")

# Step 9: Display the resized cropped image
plt.imshow(cv2.cvtColor(resized_matched_region, cv2.COLOR_BGR2RGB))
plt.title('Matched Region')
plt.axis('off')
plt.show()
