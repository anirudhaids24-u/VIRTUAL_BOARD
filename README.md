✨ Virtual Smart Board (Hand Gesture Drawing)

A real-time virtual drawing board built using Computer Vision and Hand Tracking.
Draw in the air using your fingers — no touch, no stylus, just your webcam.

---

🚀 Features

- ✋ Hand tracking using MediaPipe
- ✍️ Draw using index finger
- 🎯 Gesture-based interaction:
  - Index finger → Drawing mode
  - Index + Middle finger → Selection mode
- 🎨 Multiple colors:
  - Blue, Green, Red, Yellow
- 🧽 Eraser tool
- 🗑️ Clear canvas
- ⚡ Smooth drawing using EMA (Exponential Moving Average)
- 🎯 Jitter reduction with movement threshold
- 📸 Save drawing as image

---

🖐️ Gesture Controls

Gesture| Action
☝️ Index finger| Draw
✌️ Index + Middle| Select tools
✊ No fingers| Idle

---

🎛️ Header Tools

- Blue / Green / Red / Yellow → Draw in selected color
- Eraser → Remove strokes
- CLEAR → Reset entire canvas

---

⌨️ Keyboard Shortcuts

Q / ESC → Quit application
C       → Clear canvas
S       → Save screenshot (board_screenshot.png)

---

🏗️ Project Structure

virtual-smart-board/
│
├── virtual_smart_board.py   # Main application
└── README.md

---

⚙️ Installation

1️⃣ Clone Repository

git clone https://github.com/your-username/virtual-smart-board.git
cd virtual-smart-board

---

2️⃣ Install Dependencies

pip install opencv-python mediapipe numpy

---

▶️ Run the Application

python virtual_smart_board.py

---

🧠 How It Works

🔹 Hand Tracking

- Uses MediaPipe Hands
- Detects 21 landmarks per hand
- Tracks fingertip movement in real time

---

🔹 Gesture Recognition

- Finger positions determine mode:
  - Only index finger → drawing
  - Index + middle → selection

---

🔹 Drawing System

- Uses an OpenCV canvas (white background)
- Draws lines based on fingertip movement
- No AI recognition — pure literal drawing

---

🔹 Smoothing (EMA)

To reduce shaky lines:

new_position = α * current + (1 - α) * previous

- Produces smooth, natural strokes

---

🔹 Stability Control

- Minimum movement threshold prevents:
  - jitter
  - unwanted dots

---

🎨 Customization

Modify these parameters in the code:

STROKE_WIDTH = 8
ERASER_RADIUS = 45
SMOOTH_ALPHA = 0.40
MIN_MOVE_PX = 6

---

⚠️ Requirements

- Webcam (required)
- Python 3.8+
- Good lighting for accurate hand tracking

---

🚧 Limitations

- Supports only one hand
- Performance depends on lighting conditions
- No gesture learning (fixed gestures only)

---

🔥 Future Improvements

- 🧠 Handwriting recognition
- 🎯 Shape detection (circle, square, etc.)
- 📱 Mobile/web version
- 👥 Multi-user collaboration
- 🖼️ 3D drawing (OpenGL integration)

---

🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

---

🙌 Author

Anirudh P



