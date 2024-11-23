import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image, ImageDraw

# Load pre-trained Faster R-CNN model
segmentation_model = fasterrcnn_resnet50_fpn(pretrained=True)
segmentation_model.eval()

# Preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

# Function to limit number of objects detected and draw bounding boxes
def draw_bounding_boxes(image_path, max_objects=5):
    image_tensor, image = preprocess_image(image_path)
    
    with torch.no_grad():
        predictions = segmentation_model(image_tensor)

    # Get the predicted boxes, labels, and scores
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Sort by score in descending order (most confident detections first)
    sorted_indices = torch.argsort(scores, descending=True)

    # Limit to top N objects
    top_n_indices = sorted_indices[:max_objects]
    top_boxes = boxes[top_n_indices]
    top_labels = labels[top_n_indices]
    top_scores = scores[top_n_indices]

    # Draw bounding boxes on the image using PIL
    draw = ImageDraw.Draw(image)

    for i, (box, label, score) in enumerate(zip(top_boxes, top_labels, top_scores)):
        x1, y1, x2, y2 = box.tolist()
        label = str(label.item())  # Convert label to string (or map to class name)
        score = round(score.item(), 2)  # Round the confidence score
        
        # Draw rectangle and add label (class and score)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f'{label}: {score}', fill="red")

    return image