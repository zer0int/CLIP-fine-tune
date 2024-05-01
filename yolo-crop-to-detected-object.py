import cv2
import numpy as np
import os

# Load the YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define input and output directories
input_dir = "neurons"
output_dir = "neurons_yolo"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Loop through detections and crop
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:  # Confidence threshold
                    object_detected = True
                    class_name = classes[class_id]
                    print(f"Detected {class_name} in {filename} with confidence {confidence}")

                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate square crop dimensions
                    crop_size = min(w, h)
                    x = max(int(center_x - crop_size / 2), 0)
                    y = max(int(center_y - crop_size / 2), 0)
                    x2 = min(x + crop_size, width)
                    y2 = min(y + crop_size, height)

                    # Crop and save the image
                    cropped_img = image[y:y2, x:x2]
                    output_path = os.path.join(output_dir, "cropped_" + filename)
                    cv2.imwrite(output_path, cropped_img)

        print(f"Processed {filename}")

print("Processing complete.")
