from torchvision.transforms.v2 import RandomCrop, Resize

def random_crop(image_tensor, preserve_prop):
    """
    Randomly crops an image while preserving a specified percentage of pixels, then resizes the image back to the original size.
    
    Args:
        image_tensor: input image tensor
        preserve_percentage (float): proportion of pixels to preserve (between 0 and 1)
        
    Returns:
        Cropped image as a tensor
    """

    if not (0 < preserve_prop <= 1):
        raise ValueError("preserve_percentage must be between 0 and 1")

    # Get original image dimensions
    _, orig_height, orig_width = image_tensor.shape

    # Calculate target dimensions
    target_area = orig_width * orig_height * preserve_prop
    target_width = int((target_area * orig_width / orig_height) ** 0.5)
    target_height = int((target_area * orig_height / orig_width) ** 0.5)

    # Perform the crop
    # Crop size (height, width)
    transform = RandomCrop(size=(target_height, target_width), pad_if_needed=True, padding_mode="edge") 
    cropped_image = transform(image_tensor) 

    # rescale back to original size (height, width)
    resize_transform = Resize((orig_width, orig_height))
    resized_image = resize_transform(cropped_image)

    return resized_image