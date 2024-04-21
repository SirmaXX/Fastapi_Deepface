import cv2
from fastapi import FastAPI, WebSocket,APIRouter
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from deepface import DeepFace
import asyncio
import base64
from Routes.faces import facesroute
app = FastAPI()

app.include_router(facesroute, prefix="/faces")

