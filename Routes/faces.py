from fastapi import APIRouter, WebSocket, HTTPException, WebSocketDisconnect
import cv2
from pydantic import BaseModel
from deepface import DeepFace
import asyncio
import base64

facesroute = APIRouter(responses={404: {"description": "Not found"}})

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

class Frame(BaseModel):
    frame: bytes

# Global flag to check WebSocket status
ws_connected = False

# State to hold the last detection results
last_faces = []
last_emotion = 'none'

async def detect_emotion(websocket: WebSocket):
    global last_faces, last_emotion

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open camera")

    async def recognition_loop():
        global last_faces, last_emotion
        frame_count = 0
        recognition_interval = 10  # Perform recognition every 10 frames

        while ws_connected:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.1)
                continue

            frame_count += 1
            if frame_count % recognition_interval == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) > 0:
                    last_faces = faces
                else:
                    last_faces = []

                if len(last_faces) > 0:
                    for (x, y, w, h) in last_faces:
                        face_roi = frame[y:y + h, x:x + w]
                        result2 = DeepFace.find(face_roi, db_path='references', enforce_detection=False, silent=True)
                        last_emotion = result2[0]['identity'][0] if len(result2) > 0 and 'identity' in result2[0] and len(result2[0]['identity']) > 0 else 'none'

            if len(last_faces) > 0:
                for (x, y, w, h) in last_faces:
                    # Draw bounding box and emotion on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, last_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode()
            await websocket.send_json({'frame': frame_base64, 'emotion': last_emotion})

            await asyncio.sleep(0.033)  # Approx 30 FPS

    asyncio.create_task(recognition_loop())

    try:
        while ws_connected:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        cap.release()

@facesroute.get("/")
async def index():
    return {"message": "Welcome to Real-time Emotion Detection"}

@facesroute.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws_connected
    if ws_connected:
        await websocket.close(code=1008, reason="WebSocket is already connected")
        return

    ws_connected = True
    await websocket.accept()
    try:
        await detect_emotion(websocket)
    finally:
        ws_connected = False
