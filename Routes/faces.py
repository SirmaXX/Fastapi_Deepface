
from fastapi import APIRouter,Depends, Request,HTTPException
from flask import jsonify
from datetime import datetime
import cv2
from fastapi import FastAPI, WebSocket,APIRouter
from fastapi.responses import HTMLResponse
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

async def detect_emotion(websocket: WebSocket):
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Convert frame to JPEG format and then encode to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode()

                # Send frame bytes along with emotion through WebSocket
                await websocket.send_json({'frame': frame_base64, 'emotion': emotion})
    except WebSocketDisconnect:
        # Release video capture when WebSocket is disconnected
        cap.release()

@facesroute.get("/")
async def index():
    return {"message": "Welcome to Real-time Emotion Detection"}

@facesroute.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await detect_emotion(websocket)