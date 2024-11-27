import sys
import vpi

def check_vpi_initialization():
    # Initialize VPI
    status = vpi.vpiInit(0)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error initializing VPI: {vpi.vpiStatusToString(status)}")
        sys.exit(1)
    print("VPI initialized successfully!")

def load_image(image_path, format):
    # Load image using VPI
    image = vpi.VPIImage()
    status = vpi.vpiImageCreateFromFile(image_path, format, 0, image)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error loading image {image_path}: {vpi.vpiStatusToString(status)}")
        sys.exit(1)
    return image

def create_output_image(src_width, src_height, templ_width, templ_height):
    #Create an output image based on the input and template size
    output = vpi.VPIImage()
    status = vpi.vpiImageCreate(src_width - templ_width + 1,
                                 src_height - templ_height + 1,
                                 vpi.VPI_IMAGE_FORMAT_F32, 0, output)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error creating output image: {vpi.vpiStatusToString(status)}")
        sys.exit(1)
    return output

def perform_template_matching(input_image, templ_image, output_image, input_width, input_height):
    # Create a VPI stream
    stream = vpi.VPIStream()
    status = vpi.vpiStreamCreate(0, stream)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error creating VPI stream: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    # Create the template matching payload
    payload = vpi.VPIPayload()
    status = vpi.vpiCreateTemplateMatching(vpi.VPI_BACKEND_CUDA, input_width, input_height, payload)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error creating template matching payload: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    # Set the source (input) and template images
    status = vpi.vpiTemplateMatchingSetSourceImage(stream, vpi.VPI_BACKEND_CUDA, payload, input_image)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error setting source image: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    status = vpi.vpiTemplateMatchingSetTemplateImage(stream, vpi.VPI_BACKEND_CUDA, payload, templ_image, None)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error setting template image: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    # Submit the template matching task to the stream
    status = vpi.vpiSubmitTemplateMatching(stream, vpi.VPI_BACKEND_CUDA, payload, output_image, vpi.VPI_TEMPLATE_MATCHING_NCC)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error submitting template matching: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    # Synchronize the stream (wait until the task is done)
    status = vpi.vpiStreamSync(stream)
    if status != vpi.VPI_STATUS_SUCCESS:
        print(f"Error synchronizing stream: {vpi.vpiStatusToString(status)}")
        sys.exit(1)

    print("Template matching completed successfully!")

    # Cleanup resources
    vpi.vpiStreamDestroy(stream)

def main():
    # Initialize VPI
    check_vpi_initialization()

    # Specify input and template image paths
    input_image_path = "final/cropping_model/1 (2).png"  # Replace with your actual input image path
    template_image_path = "final/cropped_data/1.png"  # Replace with your actual template image path

    # Load input and template images
    input_image = load_image(input_image_path, vpi.VPI_IMAGE_FORMAT_U8)
    templ_image = load_image(template_image_path, vpi.VPI_IMAGE_FORMAT_U8)

    # Get image sizes
    input_width, input_height = vpi.vpiImageGetSize(input_image)
    templ_width, templ_height = vpi.vpiImageGetSize(templ_image)

    # Create an output image
    output_image = create_output_image(input_width, input_height, templ_width, templ_height)

    # Perform template matching
    perform_template_matching(input_image, templ_image, output_image, input_width, input_height)

    # Clean up images
    vpi.vpiImageDestroy(input_image)
    vpi.vpiImageDestroy(templ_image)
    vpi.vpiImageDestroy(output_image)

    # Finalize VPI
    vpi.vpiFinalize()

if __name__ == "__main__":
    main()
