from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Download model from Google Drive
def download_model():
    if not os.path.exists("best.pt"):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1YiHtOJMuN3zt1JnyOmADwemU_Sm4751D"
        r = requests.get(url)
        with open("best.pt", "wb") as f:
            f.write(r.content)

download_model()


model = YOLO("best.pt")  # Trained YOLOv8 model
reader = easyocr.Reader(['en'], gpu=False)

@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = image[y1:y2, x1:x2]
            text_result = reader.readtext(plate_img)
            text = text_result[0][1] if text_result else "Not Detected"
            return {"plate_number": text}

    return {"plate_number": "No plate found"}
