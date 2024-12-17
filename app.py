from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"uploaded_images/{file.filename}"
    os.makedirs("uploaded_images", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run YOLO inference
    results = model(file_path, conf=0.2)
    image = Image.open(file_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./roboto.ttf", 18)

    detected_objects = []

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            box = box.tolist()
            confidence = conf.item() * 100
            label = result.names[int(cls.item())]

            # Draw bounding box
            draw.rectangle(box, outline="red", width=3)
            text = f"{label} {confidence:.2f}%"
            draw.text((box[0], box[1]), text, fill="white", font=font)

            # Append detected objects
            detected_objects.append({"label": label, "confidence": confidence, "box": box})

    # Save the annotated image
    annotated_path = f"uploaded_images/annotated_{file.filename}"
    image.save(annotated_path)

    return JSONResponse(content={"objects": detected_objects, "annotated_image": annotated_path})