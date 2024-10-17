import os
from ultralytics import YOLO
import cv2
import numpy as np

# Define directories and image paths
IMAGES_DIR = os.path.join('.', 'videos')
image_path = os.path.join(IMAGES_DIR, 'unnamed-file.jpg')
image_path_out = f"{image_path[:-5]}_out.jpeg"  # Output image path

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Failed to open image: {image_path}")
    exit()

H, W, _ = image.shape  # Get image dimensions

# Load the YOLO model
model_path = os.path.join('runs', 'detect', 'train23', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.5  # Confidence threshold

class_name_dict = {
    0: 'Ambulance',
}

# Perform inference on the image
results = model(image)[0]

# Loop over detected objects
for result in results:
    if result.boxes:
        for box in result.boxes:
            conf = box.conf.item()
            cls_id = box.cls.item()  # Class ID

            if conf >= threshold:  # Only process if confidence is above threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates

                # Draw a polygon (rectangle) around the detected object
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # Four points of the polygon
                pts = cv2.polylines(image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Put class name and confidence on the image
                label = f"{class_name_dict.get(cls_id, 'Unknown')} {conf:.2f}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Class: {cls_id}, Confidence: {conf}, Box: ({x1}, {y1}, {x2}, {y2})")
    else:
        print("No detections found.")

# Save the output image with polygons
cv2.imwrite(image_path_out, image)

# Display the image (optional)
cv2.imshow('Image', image)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()

print(f"Processed image saved at: {image_path_out}")
