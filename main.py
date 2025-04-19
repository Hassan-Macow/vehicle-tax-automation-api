from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, os
import easyocr
from ultralytics import YOLO
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

def download_model():
    if not os.path.exists("best.pt"):
        print("Downloading model...")
        gdown.download("https://drive.google.com/uc?id=1YiHtOJMuN3zt1JnyOmADwemU_Sm4751D", "best.pt", quiet=False)

download_model()
reader = easyocr.Reader(['en'], gpu=False)

@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    model = YOLO("best.pt")
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = image[y1:y2, x1:x2]
            result = reader.readtext(plate_img)
            text = result[0][1] if result else "Not Detected"
            return {"plate_number": text}

    return {"plate_number": "No plate found"}
