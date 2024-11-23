from PIL import Image
from vcd_add_noise import add_diffusion_noise

# Define the path to the image file
image_path = "experiments/data/coco/COCO_val2014_000000000042.jpg"  # Replace with the path to your image

try:
    # Open the image
    img = Image.open(image_path)

    # Display the image
    img.show()

    noise_img = add_diffusion_noise(img, 500)

    # Print image details
    print(f"Image format: {img.format}")
    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")

except FileNotFoundError:
    print(f"Error: The file at {image_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
