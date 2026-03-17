from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

process = None

@app.get("/")
def home():
    return {"message": "Virtual Smart Board Backend Running"}

@app.get("/start")
def start_board():
    global process
    if process is None:
        process = subprocess.Popen(["python", "backend/virtual_smart_board.py"])
        return {"status": "Board Started"}
    return {"status": "Already Running"}

@app.get("/stop")
def stop_board():
    global process
    if process:
        process.terminate()
        process = None
        return {"status": "Board Stopped"}
    return {"status": "Not Running"}
