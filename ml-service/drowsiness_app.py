import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------
# CONFIG (tune these thresholds)
# -----------------------------
CONF_THRES = 0.35

# Time thresholds (seconds)
LOOK_AWAY_TIME = 2.0          # looking away sustained
PHONE_TIME = 3.0              # phone use sustained
DROWSY_TIME = 2.0             # eyes closed sustained -> drowsy
SLEEP_TIME = 4.0              # eyes closed longer -> sleeping alarm

# Yawn logic
YAWN_WINDOW = 30.0            # seconds window to count yawns
YAWN_COUNT_WARN = 3           # yawns in window to warn "take a rest"

# Alarm settings
ALARM_COOLDOWN = 3.0          # seconds between alarm triggers

# --------------------------------
# Optional: simple sound alarm
# --------------------------------
def beep():
    # Cross-platform-ish fallback using OpenCV tone is not built-in;
    # easiest: Windows winsound; else print.
    try:
        import winsound
        winsound.Beep(2000, 600)
    except Exception:
        print("[ALARM] BEEP (install sound lib or use winsound on Windows)")

# --------------------------------
# Helper: choose best detection per class
# --------------------------------
def best_box_for_class(dets, class_id):
    """Return the best (highest conf) bbox for class_id: (x1,y1,x2,y2,conf) or None."""
    best = None
    for (x1, y1, x2, y2, conf, cls) in dets:
        if int(cls) == int(class_id):
            if best is None or conf > best,[object Object],:
                best = (x1, y1, x2, y2, conf)
    return best

def now():
    return time.time()

# --------------------------------
# MAIN
# --------------------------------
def run(video_source=0, model_path="my_model/my_model.pt", save_output=True):
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    # For saving output video with overlays
    writer = None
    out_path = "output_drowsiness_demo.mp4"

    # State timers
    state_start = defaultdict(lambda: None)  # state -> start_time
    last_alarm_time = 0.0

    # yawn timestamps
    yawn_times = deque()

    # class name mapping (ensure matches your data.yaml)
    # You can also read from model.names
    names = model.names

    # Utility: start/stop timers
    def set_state(state, active):
        if active:
            if state_start[state] is None:
                state_start[state] = now()
        else:
            state_start[state] = None

    def state_duration(state):
        t0 = state_start[state]
        return 0.0 if t0 is None else (now() - t0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        # YOLO inference
        results = model.predict(frame, conf=CONF_THRES, verbose=False, device=device)

        dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy()
            for (b, c, k) in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = b
                dets.append((float(x1), float(y1), float(x2), float(y2), float(c), float(k)))

        # Flags from detections
        # IMPORTANT: update ids if your yaml order differs
        CLASS_PHONE = 0
        CLASS_YAWN = 1
        CLASS_EYES_CLOSED = 2
        CLASS_FACE_FRONT = 3
        CLASS_FACE_LEFT = 4
        CLASS_FACE_RIGHT = 5
        CLASS_FACE_DOWN = 6

        phone_box = best_box_for_class(dets, CLASS_PHONE)
        yawn_box = best_box_for_class(dets, CLASS_YAWN)
        eyes_closed_box = best_box_for_class(dets, CLASS_EYES_CLOSED)

        face_front = best_box_for_class(dets, CLASS_FACE_FRONT) is not None
        face_left = best_box_for_class(dets, CLASS_FACE_LEFT) is not None
        face_right = best_box_for_class(dets, CLASS_FACE_RIGHT) is not None
        face_down = best_box_for_class(dets, CLASS_FACE_DOWN) is not None

        looking_away = (face_left or face_right or face_down) and (not face_front)

        on_phone = phone_box is not None
        yawning = yawn_box is not None
        eyes_closed = eyes_closed_box is not None

        # Update timers
        set_state("looking_away", looking_away)
        set_state("on_phone", on_phone)
        set_state("eyes_closed", eyes_closed)

        # Yawn events (count yawns within rolling window)
        # Add timestamp when yawn appears (with simple debounce)
        if yawning:
            # Debounce: only add if last yawn was > 1.0s ago
            if len(yawn_times) == 0 or (now() - yawn_times[-1]) > 1.0:
                yawn_times.append(now())

        # Drop old yawns outside window
        while len(yawn_times) > 0 and (now() - yawn_times,[object Object],) > YAWN_WINDOW:
            yawn_times.popleft()

        yawn_count = len(yawn_times)

        # Decide warnings
        warnings = []

        if state_duration("looking_away") >= LOOK_AWAY_TIME:
            warnings.append("LOOKING AWAY: Please focus on road")

        if state_duration("on_phone") >= PHONE_TIME:
            warnings.append("PHONE USE: Put the phone away")

        drowsy = state_duration("eyes_closed") >= DROWSY_TIME
        sleeping = state_duration("eyes_closed") >= SLEEP_TIME

        if yawn_count >= YAWN_COUNT_WARN:
            warnings.append(f"FATIGUE: {yawn_count} yawns/{int(YAWN_WINDOW)}s - Take a rest")

        if drowsy and not sleeping:
            warnings.append("DROWSY: Eyes closed - Stay alert")

        if sleeping:
            warnings.append("SLEEPING: WAKE UP NOW!")
            if (now() - last_alarm_time) >= ALARM_COOLDOWN:
                beep()
                last_alarm_time = now()

        # Draw detections
        for (x1, y1, x2, y2, conf, cls) in dets:
            cls = int(cls)
            label = f"{names.get(cls, str(cls))} {conf:.2f}"
            color = (0, 255, 0)

            # Highlight risky classes
            if cls in [CLASS_EYES_CLOSED, CLASS_YAWN, CLASS_PHONE, CLASS_FACE_LEFT, CLASS_FACE_RIGHT, CLASS_FACE_DOWN]:
                color = (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, label, (int(x1), max(0, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Draw status panel
        panel_y = 30
        status_lines = [
            f"looking_away: {looking_away} ({state_duration('looking_away'):.1f}s)",
            f"on_phone: {on_phone} ({state_duration('on_phone'):.1f}s)",
            f"eyes_closed: {eyes_closed} ({state_duration('eyes_closed'):.1f}s)",
            f"yawns(last {int(YAWN_WINDOW)}s): {yawn_count}",
        ]
        for line in status_lines:
            cv2.putText(frame, line, (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            panel_y += 24

        # Draw warnings
        wy = panel_y + 10
        for wtxt in warnings[:4]:
            cv2.putText(frame, wtxt, (10, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            wy += 30

        # Init writer if needed
        if save_output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 1:
                fps = 25
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Driver Drowsiness (YOLOv10)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved demo video to: {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 0 for webcam, or set a path like "test_video.mp4"
    run(video_source=0, model_path="my_model/my_model.pt", save_output=True)