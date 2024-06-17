import cv2
from fastapi import FastAPI, WebSocket,APIRouter
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from deepface import DeepFace
import asyncio
import base64
from Routes.faces import facesroute

app = FastAPI()
# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(facesroute, prefix="/faces")

