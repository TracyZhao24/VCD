# Better object detection, but blurs entire box
import cv2
from PIL import Image

def gaussian_blur_yolo_box(image_path, model, blur_width=35):
    # Load image
    image = cv2.imread(image_path)
    src_img = image.copy()

    # Perform inference
    results = model(image)

    # Display results
    results.show()  # This shows the image with detections

    # Extract bounding boxes and confidence scores
    boxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes (x1, y1, x2, y2)
    labels = results.names  # Get the label names
    confidences = results.xywh[0][:, 4].cpu().numpy()  # Confidence scores (the 5th column)

    # Create a copy of the image to apply blur
    blurred_image = src_img.copy()
    # cv2_imshow(blurred_image)

    # Iterate over each detected object
    for box, confidence in zip(boxes, confidences):
        if confidence > 0.0:  # Only consider detections with high confidence
            # Get coordinates of the bounding box
            
            # print(box)
            x1, y1, x2, y2, _, _ = map(int, box)

            # Extract the object region
            object_region = src_img[y1:y2, x1:x2]

            # Apply Gaussian blur to the object region
            blurred_region = cv2.GaussianBlur(object_region, (blur_width, blur_width), 0)

            # Replace the object region in the blurred image with the blurred region
            blurred_image[y1:y2, x1:x2] = blurred_region
    
    rgb_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # Convert the NumPy array to a PIL image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

