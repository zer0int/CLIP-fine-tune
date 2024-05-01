import cv2
import numpy as np
import os

# Load the YOLO model
net = cv2.dnn.readNet("yolov7.weights", "yolov7.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define input and output directories
input_dir = "COCO_train/train2014"
output_dir = "COCO_yolo"

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(valid_extensions):
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
                    class_name = classes[class_id]
                    print(f"Detected {class_name} in {filename} with confidence {confidence}")

                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Determine square crop size based on detected object, ensuring it's at least 224
                    square_crop_size = max(max(w, h), 224)

                    # Calculate the top-left corner of the crop
                    x = int(center_x - square_crop_size / 2)
                    y = int(center_y - square_crop_size / 2)

                    # Ensure the crop does not exceed image boundaries and adjust if necessary
                    # Adjust the X coordinate and width
                    if x < 0:
                        x = 0
                    elif x + square_crop_size > width:
                        x = width - square_crop_size

                    # Adjust the Y coordinate and height
                    if y < 0:
                        y = 0
                    elif y + square_crop_size > height:
                        y = height - square_crop_size

                    # Now we have ensured the crop is within image boundaries and at least 224x224
                    cropped_img = image[y:y + square_crop_size, x:x + square_crop_size]
                    output_path = os.path.join(output_dir, class_name + "_" + filename)
                    cv2.imwrite(output_path, cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])



        print(f"Processed {filename}")

print("Processing complete.")
