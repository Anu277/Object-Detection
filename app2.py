from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
import torch
import matplotlib.pyplot as plt
import random  # To randomize the confidence score

# Load YOLOv8 model
model = YOLO("./models/yolov8n.pt")  # Load the YOLOv8n model from the file

# Function to draw detections on the image
def draw_detections(image_path, results):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = font = ImageFont.truetype(r"D:\Python Projects\Object Detection\backend\static\roboto.ttf", 18)

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            box = box.tolist()  # Convert bounding box tensor to a list
            confidence = conf.item() * 100  # Confidence score
            label = result.names[int(cls.item())]  # Class label

            # Draw bounding box
            draw.rectangle(box, outline="red", width=3)
            text = f"{label} {confidence:.2f}%"
            draw.text((box[0], box[1]),  text, fill="white", font=font)

    image.show()  # Display the image locally

# Provide the path to your local image
image_path = r"D:\Python Projects\Object Detection\backend\img.jpg"  # Replace with your image path

# Run YOLOv8 inference
results = model(image_path, conf=0.2)  # Set confidence threshold
draw_detections(image_path, results)



# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# from PIL import Image, ImageDraw, ImageFont
# import io
# import torch
# import base64
# import numpy as np

# # Initialize the Flask app
# app = Flask(__name__)

# # Load YOLOv8 model
# model = YOLO("models/yolov8n.pt")

# # Function to process image and detect objects
# def process_image(image_bytes):
#     # Convert byte data to image
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
#     # Run YOLOv8 inference
#     results = model(image)
    
#     # Drawing detections on the image
#     output_image = draw_detections(image, results)
    
#     # Save the output image as byte array
#     img_byte_arr = io.BytesIO()
#     output_image.save(img_byte_arr, format='PNG')
#     img_byte_arr.seek(0)
    
#     # Return the image as base64 string
#     return base64.b64encode(img_byte_arr.read()).decode('utf-8')

# # Function to draw bounding boxes on image
# def draw_detections(image, results):
#     draw = ImageDraw.Draw(image)
#     font = ImageFont.truetype("./static/roboto.ttf", 18)
    
#     for result in results:
#         for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
#             box = box.tolist()
#             confidence = conf.item() * 100
#             label = result.names[int(cls.item())]
            
#             # Draw bounding box
#             draw.rectangle(box, outline="red", width=3)
#             text = f"{label} {confidence:.2f}%"
#             draw.text((box[0], box[1]), text, fill="white", font=font)
    
#     return image

# # API endpoint to handle image upload and detection
# @app.route('/detect', methods=['POST'])
# def detect_objects():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
    
#     image_bytes = file.read()
#     result_image = process_image(image_bytes)
    
#     return jsonify({"image": result_image})

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
