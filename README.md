# ✨ Virtual Smart Board + Web Interface

An AI-powered **Virtual Smart Board** that allows users to draw using hand gestures and interact with it through a web interface.

This project combines:

* 🎥 Computer Vision (MediaPipe + OpenCV)
* 🧠 Gesture Recognition
* 🌐 Web-based frontend
* ⚙️ Python backend (FastAPI)

---

# 🚀 Features

## 🖐️ Smart Board (Computer Vision)

* Draw using index finger
* Select tools using index + middle finger
* Smooth drawing with EMA filtering
* Eraser + clear canvas
* Save drawing as image

## 🌐 Web Interface

* Upload & view board output
* Control backend from browser
* Extendable for AI + cloud features

---

# 🏗️ Project Architecture

```text
Frontend (HTML/CSS/JS)
        │
        ▼
FastAPI Backend
        │
        ▼
Virtual Smart Board (OpenCV + MediaPipe)
        │
        ▼
Camera Input → Gesture → Drawing
```

---

# 📂 Project Structure

```text
virtual-smart-board/
│
├── backend/
│   ├── main.py                # FastAPI server
│   ├── virtual_smart_board.py # Core CV logic
│
├── frontend/
│   └── index.html            # UI interface
│
├── data/
│   └── board_screenshot.png
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/virtual-smart-board.git
cd virtual-smart-board
```

---

## 2️⃣ Install Python Dependencies

```bash
pip install fastapi uvicorn opencv-python mediapipe numpy
```

---

# 🧠 Backend Setup (FastAPI)

## 🔹 Create `main.py`

```python
from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Virtual Smart Board Backend Running"}

@app.get("/start")
def start_board():
    subprocess.Popen(["python", "virtual_smart_board.py"])
    return {"status": "Board Started"}

@app.get("/stop")
def stop_board():
    return {"status": "Stop manually (press Q in window)"}
```

---

## ▶️ Run Backend

```bash
uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

# 🌐 Frontend Setup

## 🔹 Create `frontend/index.html`

```html
<!DOCTYPE html>
<html>
<head>
  <title>Virtual Smart Board</title>
</head>
<body>

<h1>🧠 Virtual Smart Board</h1>

<button onclick="startBoard()">Start Board</button>
<button onclick="stopBoard()">Stop Board</button>

<script>
async function startBoard() {
  await fetch("http://127.0.0.1:8000/start");
  alert("Board Started!");
}

async function stopBoard() {
  await fetch("http://127.0.0.1:8000/stop");
}
</script>

</body>
</html>
```

---

## ▶️ Run Frontend

Option 1:

```bash
open index.html
```

Option 2 (recommended):

```bash
python -m http.server 5500
```

Open:

```
http://localhost:5500
```

---

# 🎮 How to Use

## 🖐️ Gestures

| Gesture           | Action |
| ----------------- | ------ |
| ☝️ Index finger   | Draw   |
| ✌️ Index + Middle | Select |
| ✊ No fingers      | Idle   |

---

## 🎨 Tools

* Blue / Green / Red / Yellow → Drawing colors
* Eraser → Remove strokes
* CLEAR → Reset board

---

## ⌨️ Keyboard Shortcuts

```text
Q / ESC → Quit
C       → Clear canvas
S       → Save screenshot
```

---

# 🧠 How It Works

## 1. Hand Tracking

* MediaPipe detects 21 landmarks
* Tracks fingertip position

## 2. Gesture Detection

* Finger combinations → modes

## 3. Drawing Engine

* OpenCV canvas
* Line rendering with smoothing

## 4. Backend Integration

* FastAPI triggers Python CV script
* Frontend sends HTTP requests

---

# 🔧 Configuration

Modify in `virtual_smart_board.py`:

```python
STROKE_WIDTH = 8
ERASER_RADIUS = 45
SMOOTH_ALPHA = 0.40
MIN_MOVE_PX = 6
```

---

# ⚠️ Requirements

* Webcam required
* Good lighting
* Python 3.8+

---

# 🚧 Limitations

* Single-hand tracking only
* Manual stop required
* No browser video streaming yet

---

# 🔥 Future Improvements

* 📡 Live video stream in browser
* 🧠 AI handwriting recognition
* ☁️ Cloud sync
* 👥 Multi-user collaboration
* 🎯 Gesture customization

---

# 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* FastAPI
* HTML / JavaScript

---

# 🙌 Author

**Anirudh P**

---

⭐ Star this repo if you like it!
