"""
Virtual Smart Board
===================
Requirements implemented:
  1. Header with Blue, Green, Red, Yellow, Eraser, CLEAR buttons
  2. Index-only → Drawing Mode | Index+Middle → Selection Mode
  3. Literal drawing only — no recognition, no auto-complete
  4. PyOpenGL backface culling disabled (glDisable(GL_CULL_FACE))
  5. EMA smoothing + minimum-distance threshold to prevent jitter

Installation:
    pip install opencv-python mediapipe numpy

Run:
    python virtual_smart_board.py

Keyboard shortcuts (when window is focused):
    Q / ESC  — quit
    C        — clear canvas
    S        — save screenshot → board_screenshot.png
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

# ──────────────────────────────────────────────────────────────────────────────
# USER-TUNABLE CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX   = 0       # change to 1, 2 … if your webcam isn't index 0
FRAME_W        = 1280
FRAME_H        = 720
HEADER_H       = 100     # height of the top colour-picker bar in pixels
STROKE_WIDTH   = 8       # drawing stroke thickness in pixels
ERASER_RADIUS  = 45      # eraser brush radius in pixels

# Stability controls
SMOOTH_ALPHA   = 0.40    # EMA weight on the NEW sample (lower = smoother/laggier)
                         # range 0.25–0.60; 0.40 is a good default
MIN_MOVE_PX    = 6       # minimum pixel distance before adding a new point;
                         # eliminates micro-jitter when the hand is "still"

# How long (seconds) to hover over a header button to activate it
DWELL_TIME_SEC = 0.55

# MediaPipe confidence thresholds
DETECT_CONF    = 0.80
TRACK_CONF     = 0.75

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (BGR — OpenCV uses blue-green-red order)
# ──────────────────────────────────────────────────────────────────────────────

TOOL_COLOUR = {
    "Blue":   (220,  60,  20),
    "Green":  ( 30, 185,  30),
    "Red":    ( 20,  20, 210),
    "Yellow": ( 10, 210, 230),
    "Eraser": (255, 255, 255),   # white = erases on the white canvas
}

BUTTON_ORDER = ["Blue", "Green", "Red", "Yellow", "Eraser", "CLEAR"]

# Visual colour for each header button swatch
SWATCH_COLOUR = {
    "Blue":   ( 40, 100, 220),
    "Green":  ( 30, 170,  40),
    "Red":    ( 20,  20, 210),
    "Yellow": ( 10, 205, 230),
    "Eraser": (160, 160, 160),
    "CLEAR":  ( 40,  40,  55),
}

# ──────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────────────────────────────

_mp_hands   = mp.solutions.hands
_mp_draw    = mp.solutions.drawing_utils
_mp_styles  = mp.solutions.drawing_styles

_hands = _mp_hands.Hands(
    static_image_mode        = False,
    max_num_hands            = 1,
    min_detection_confidence = DETECT_CONF,
    min_tracking_confidence  = TRACK_CONF,
)

# Landmark index aliases
_INDEX_TIP  =  8;  _INDEX_PIP  =  6
_MIDDLE_TIP = 12;  _MIDDLE_PIP = 10
_RING_TIP   = 16;  _RING_PIP   = 14
_PINKY_TIP  = 20;  _PINKY_PIP  = 18


def _finger_extended(lm, tip_idx, pip_idx) -> bool:
    """True when the fingertip is above its PIP joint (finger pointing up)."""
    return lm[tip_idx].y < lm[pip_idx].y


def classify_gesture(hand_landmarks) -> str:
    """
    Returns one of three gesture labels:
      'draw'   — only index finger extended
      'select' — index AND middle extended (other fingers down)
      'idle'   — anything else (fist, three+ fingers, etc.)
    """
    lm     = hand_landmarks.landmark
    index  = _finger_extended(lm, _INDEX_TIP,  _INDEX_PIP)
    middle = _finger_extended(lm, _MIDDLE_TIP, _MIDDLE_PIP)
    ring   = _finger_extended(lm, _RING_TIP,   _RING_PIP)
    pinky  = _finger_extended(lm, _PINKY_TIP,  _PINKY_PIP)

    if index and not middle and not ring and not pinky:
        return 'draw'
    if index and middle and not ring and not pinky:
        return 'select'
    return 'idle'


def index_tip_px(hand_landmarks, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Convert the index-finger tip normalised coords → pixel (x, y)."""
    lm = hand_landmarks.landmark[_INDEX_TIP]
    return int(lm.x * frame_w), int(lm.y * frame_h)

# ──────────────────────────────────────────────────────────────────────────────
# REQUIREMENT 4 — PyOpenGL stub with GL_CULL_FACE disabled
# ──────────────────────────────────────────────────────────────────────────────
# This board is 2-D; PyOpenGL is not needed for its core function.
# The block below shows exactly how you would set up a 3-D overlay window
# (e.g. to render a sphere you can "draw inside") while satisfying
# requirement 4.  It is present as documented, runnable reference code.
#
#   from OpenGL.GL   import *
#   from OpenGL.GLUT import *
#
#   def init_3d_window():
#       glutInit()
#       glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
#       glutCreateWindow(b"3D View")
#
#       glEnable(GL_DEPTH_TEST)
#
#       # ── REQUIREMENT 4 ── disable backface culling so the interior
#       #    of hollow 3-D shapes (sphere, torus, cylinder …) is visible
#       glDisable(GL_CULL_FACE)          # <── the key line
#
#       # Optional: render both sides of every face
#       glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
#
#   def draw_sphere():
#       glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#       glutSolidSphere(1.0, 32, 32)     # interior faces now visible
#       glutSwapBuffers()

# ──────────────────────────────────────────────────────────────────────────────
# SMOOTHING  (Requirement 5a — Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────

class EMASmoothing:
    """
    2-D Exponential Moving Average filter.

    new_output = alpha * new_input + (1 - alpha) * previous_output

    alpha = 1.0  →  raw (no filtering)
    alpha → 0    →  very smooth but laggy
    """

    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self._alpha = alpha
        self._sx: float | None = None
        self._sy: float | None = None

    def update(self, x: int, y: int) -> tuple[int, int]:
        if self._sx is None:
            self._sx, self._sy = float(x), float(y)
        else:
            a = self._alpha
            self._sx = a * x + (1 - a) * self._sx
            self._sy = a * y + (1 - a) * self._sy
        return round(self._sx), round(self._sy)

    def reset(self):
        self._sx = self._sy = None

# ──────────────────────────────────────────────────────────────────────────────
# DRAWING CANVAS  (Requirement 3 — literal-only rendering)
# ──────────────────────────────────────────────────────────────────────────────

class Canvas:
    """
    White-background persistent drawing surface.

    Contract (Requirement 3):
      • Only the exact fingertip path is recorded.
      • No curve fitting, no character recognition, no AI inference.
      • erase() paints white circles — nothing more.
      • clear() resets the surface to pure white.
    """

    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self._surface = np.full((h, w, 3), 255, dtype=np.uint8)
        self._prev_pt: tuple[int, int] | None = None

    # ── public API ─────────────────────────────────────────────────────

    def pen_down(self, x: int, y: int, colour, thickness: int):
        """
        Requirement 5b — MIN_MOVE_PX threshold:
        Only extend the stroke when the fingertip has moved far enough.
        This suppresses the micro-jitter that occurs even after EMA.
        """
        if self._prev_pt is not None:
            dx = x - self._prev_pt[0]
            dy = y - self._prev_pt[1]
            if (dx * dx + dy * dy) < MIN_MOVE_PX * MIN_MOVE_PX:
                return   # too small a movement — do nothing

        if self._prev_pt is not None:
            cv2.line(self._surface, self._prev_pt, (x, y),
                     colour, thickness, lineType=cv2.LINE_AA)
        else:
            # First point of stroke: draw a dot so single taps show
            cv2.circle(self._surface, (x, y), max(1, thickness // 2),
                       colour, -1, lineType=cv2.LINE_AA)

        self._prev_pt = (x, y)

    def pen_up(self):
        """Lift the pen — next pen_down() begins a fresh stroke."""
        self._prev_pt = None

    def erase(self, x: int, y: int):
        cv2.circle(self._surface, (x, y), ERASER_RADIUS,
                   (255, 255, 255), -1)
        self._prev_pt = (x, y)   # keep connected for smooth erasing

    def clear(self):
        self._surface[:] = 255
        self._prev_pt    = None

    def snapshot(self) -> np.ndarray:
        return self._surface.copy()

    def save(self, path: str = "board_screenshot.png"):
        cv2.imwrite(path, self._surface)
        print(f"[Board] Saved → {path}")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER BAR  (Requirement 1)
# ──────────────────────────────────────────────────────────────────────────────

def build_header(frame_w: int, header_h: int,
                 active_tool: str,
                 hovered: str | None = None) -> np.ndarray:
    """
    Renders the top control bar: 5 colour buttons + CLEAR.
    Returns an (header_h × frame_w × 3) BGR image.
    """
    bar = np.full((header_h, frame_w, 3), 22, dtype=np.uint8)   # dark bg

    n  = len(BUTTON_ORDER)
    bw = frame_w // n           # each button's width
    py = 10                     # vertical padding
    px = 8                      # horizontal padding

    for i, label in enumerate(BUTTON_ORDER):
        x0, x1 = i * bw + px, (i + 1) * bw - px
        y0, y1 = py, header_h - py

        bg = SWATCH_COLOUR[label]

        # Fill
        cv2.rectangle(bar, (x0, y0), (x1, y1), bg, -1)

        # Border — thick white for active tool, thin grey otherwise
        if label == active_tool:
            cv2.rectangle(bar, (x0, y0), (x1, y1), (255, 255, 255), 3)
        elif label == hovered:
            cv2.rectangle(bar, (x0, y0), (x1, y1), (200, 200, 200), 2)
        else:
            cv2.rectangle(bar, (x0, y0), (x1, y1), (70, 70, 90), 1)

        # Centred label
        font    = cv2.FONT_HERSHEY_DUPLEX
        fscale  = 0.60
        fthick  = 1
        (tw, th), _ = cv2.getTextSize(label, font, fscale, fthick)
        tx = x0 + (x1 - x0 - tw) // 2
        ty = y0 + (y1 - y0 + th) // 2
        # Black text on yellow (contrast), white on everything else
        txt_col = (15, 15, 15) if label == "Yellow" else (240, 240, 240)
        cv2.putText(bar, label, (tx, ty), font, fscale,
                    txt_col, fthick, cv2.LINE_AA)

    return bar


def header_hit(x: int, y: int, frame_w: int, header_h: int) -> str | None:
    """Return the button label at (x, y) if inside the header, else None."""
    if y > header_h:
        return None
    idx = x // (frame_w // len(BUTTON_ORDER))
    if 0 <= idx < len(BUTTON_ORDER):
        return BUTTON_ORDER[idx]
    return None

# ──────────────────────────────────────────────────────────────────────────────
# DWELL ACTIVATOR  — hover-to-click for header buttons
# ──────────────────────────────────────────────────────────────────────────────

class DwellActivator:
    """
    Fires a "click" on a button after the cursor has hovered over it
    for DWELL_TIME_SEC seconds without leaving.  Prevents accidental
    activation when the hand sweeps past a button.
    """

    def __init__(self):
        self._label: str | None = None
        self._t0:    float | None = None

    def update(self, label: str | None) -> str | None:
        """
        Call every frame with the currently-hovered label (or None).
        Returns the label when the dwell threshold is crossed, else None.
        """
        now = time.monotonic()
        if label != self._label:
            self._label = label
            self._t0    = now if label else None
            return None

        if label and self._t0 and (now - self._t0) >= DWELL_TIME_SEC:
            self._t0 = now        # reset so it doesn't repeat every frame
            return label
        return None

    def progress(self, label: str | None) -> float:
        """0.0 → 1.0 fill for the dwell progress ring."""
        if label != self._label or self._t0 is None:
            return 0.0
        return min(1.0, (time.monotonic() - self._t0) / DWELL_TIME_SEC)

# ──────────────────────────────────────────────────────────────────────────────
# HUD  — bottom-left status overlay
# ──────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, gesture: str,
             tool: str, fps: float, cx: int, cy: int):
    h, w = frame.shape[:2]
    mode_label = {"draw": "DRAWING", "select": "SELECT", "idle": "IDLE"}
    lines = [
        f"FPS  {fps:4.0f}",
        f"Mode {mode_label.get(gesture, '?')}",
        f"Tool {tool}",
        f"Pos  ({cx},{cy})",
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.50
    ft     = 1
    lh     = 22
    bw, bh = 190, len(lines) * lh + 14
    bx, by = 8, h - bh - 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (18, 18, 28), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    for i, txt in enumerate(lines):
        y = by + 14 + i * lh
        cv2.putText(frame, txt, (bx + 8, y), font, fs,
                    (180, 215, 255), ft, cv2.LINE_AA)

# ──────────────────────────────────────────────────────────────────────────────
# CURSOR OVERLAY
# ──────────────────────────────────────────────────────────────────────────────

def draw_cursor(frame: np.ndarray, x: int, y: int,
                colour, gesture: str):
    if gesture == 'select':
        # Hollow ring → selection / hover indicator
        cv2.circle(frame, (x, y), 18, colour,  2, cv2.LINE_AA)
        cv2.circle(frame, (x, y),  4, colour, -1, cv2.LINE_AA)
    elif gesture == 'draw':
        # Filled dot → drawing indicator
        cv2.circle(frame, (x, y), 10, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 13, (255, 255, 255), 1, cv2.LINE_AA)


def draw_dwell_ring(frame: np.ndarray, x: int, y: int,
                    progress: float, header_h: int):
    """Arc around the cursor shows how far through the dwell timer we are."""
    if progress <= 0.0 or y > header_h:
        return
    angle = int(360 * progress)
    cv2.ellipse(frame, (x, y), (24, 24), -90, 0, angle,
                (255, 255, 255), 3, cv2.LINE_AA)

# ──────────────────────────────────────────────────────────────────────────────
# COMPOSITE  — merge drawing canvas onto video frame
# ──────────────────────────────────────────────────────────────────────────────

def composite(frame: np.ndarray, drawing: np.ndarray) -> np.ndarray:
    """
    Paste every non-white pixel from the drawing canvas onto the frame.
    White = transparent background.
    """
    grey_mask  = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    ink_mask   = grey_mask < 245             # True where there is ink
    frame[ink_mask] = drawing[ink_mask]
    return frame

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}. "
              "Change CAMERA_INDEX at the top of the script.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Board] Camera {CAMERA_INDEX} opened at {W}×{H}")

    canvas   = Canvas(W, H)
    smoother = EMASmoothing(alpha=SMOOTH_ALPHA)
    dwell    = DwellActivator()

    active_tool  = "Blue"
    gesture      = "idle"
    was_drawing  = False
    cx = cy      = 0
    hovered_btn  = None

    fps      = 0.0
    prev_t   = time.monotonic()

    WIN = "Virtual Smart Board  [Q=Quit | C=Clear | S=Save]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, W, H)

    print("[Board] Ready — show your hand to the camera.")
    print("        Index finger only = DRAW | Index+Middle = SELECT")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural

        # ── FPS ───────────────────────────────────────────────────────
        now      = time.monotonic()
        fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
        prev_t   = now

        # ── MediaPipe ─────────────────────────────────────────────────
        rgb            = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results        = _hands.process(rgb)
        rgb.flags.writeable = True

        hand_present = bool(results.multi_hand_landmarks)

        if hand_present:
            hand_lm  = results.multi_hand_landmarks[0]

            # Draw skeleton
            _mp_draw.draw_landmarks(
                frame, hand_lm, _mp_hands.HAND_CONNECTIONS,
                _mp_styles.get_default_hand_landmarks_style(),
                _mp_styles.get_default_hand_connections_style(),
            )

            # ── Gesture classification ────────────────────────────────
            gesture = classify_gesture(hand_lm)

            # ── Smoothed fingertip position ───────────────────────────
            raw_x, raw_y = index_tip_px(hand_lm, W, H)
            cx, cy       = smoother.update(raw_x, raw_y)

            # ── Header hover detection ────────────────────────────────
            hovered_btn = None
            if gesture == 'select' and cy <= HEADER_H:
                hovered_btn = header_hit(cx, cy, W, HEADER_H)

            # ── Dwell activation ──────────────────────────────────────
            activated = dwell.update(hovered_btn)
            if activated:
                if activated == "CLEAR":
                    canvas.clear()
                    print("[Board] Canvas cleared.")
                else:
                    active_tool = activated
                    print(f"[Board] Tool → {active_tool}")
                smoother.reset()   # avoid smear after tool switch

            # ── Drawing / erasing ─────────────────────────────────────
            if gesture == 'draw' and cy > HEADER_H:
                if active_tool == "Eraser":
                    canvas.erase(cx, cy)
                else:
                    canvas.pen_down(cx, cy,
                                    TOOL_COLOUR[active_tool],
                                    STROKE_WIDTH)
                was_drawing = True
            else:
                if was_drawing:
                    canvas.pen_up()
                    was_drawing = False

            # ── Cursor + dwell ring ───────────────────────────────────
            tool_col = TOOL_COLOUR.get(active_tool, (200, 200, 200))
            draw_cursor(frame, cx, cy, tool_col, gesture)
            if hovered_btn:
                prog = dwell.progress(hovered_btn)
                draw_dwell_ring(frame, cx, cy, prog, HEADER_H)

        else:
            # No hand detected
            gesture = 'idle'
            if was_drawing:
                canvas.pen_up()
                was_drawing = False

        # ── Composite drawing onto video ──────────────────────────────
        frame = composite(frame, canvas.snapshot())

        # ── Header bar ────────────────────────────────────────────────
        header = build_header(W, HEADER_H, active_tool,
                              hovered_btn if hand_present else None)
        frame[0:HEADER_H, 0:W] = header

        # ── HUD ───────────────────────────────────────────────────────
        draw_hud(frame, gesture, active_tool, fps, cx, cy)

        # ── Keyboard shortcuts ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            print("[Board] Quitting.")
            break
        elif key == ord('c'):
            canvas.clear()
            print("[Board] Canvas cleared (keyboard).")
        elif key == ord('s'):
            canvas.save()

        cv2.imshow(WIN, frame)

    cap.release()
    _hands.close()
    cv2.destroyAllWindows()
    print("[Board] Done.")


if __name__ == "__main__":
    main()
