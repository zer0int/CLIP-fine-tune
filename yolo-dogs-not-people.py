import cv2
import numpy as np
import os
import shutil

# Load the YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define input and output directories
input_dir = "wlddog-raw"
output_dir = "wlddog-yolo"
delcandidates_dir = "delcandidates"

# Create the output directories if they don't exist
for dir in [output_dir, delcandidates_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        person_detected = False
        dog_detected = False

        # Loop through detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                class_name = classes[class_id]

                if class_name == "person" and confidence > 0.7:  # Adjust confidence threshold as needed
                    person_detected = True
                    break

                if class_name == "dog" and confidence > 0.05:
                    dog_detected = True
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Adjust for square crop and minimum size
                    crop_size = max(min(w, h), 336)
                    x = max(center_x - crop_size // 2, 0)
                    y = max(center_y - crop_size // 2, 0)
                    x2 = min(x + crop_size, width)
                    y2 = min(y + crop_size, height)

                    # Crop and save the image
                    cropped_img = image[y:y2, x:x2]
                    output_path = os.path.join(output_dir, "cropped_" + filename)
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Cropped and saved {filename}")

            if person_detected:
                shutil.move(image_path, os.path.join(delcandidates_dir, filename))
                print(f"Moved {filename} to delcandidates due to person detection")
                break

        if not person_detected and not dog_detected:
            print(f"No dog detected in {filename}")

print("Processing complete.")
