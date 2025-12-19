import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Argument parser
parser = argparse.ArgumentParser(description="YOLO Object Detection Script")

parser.add_argument(
    '--model',
    required=True,
    help='Path to YOLO model file, for example: /Users/namuun/AI_engineer/Nov_29/my_model/train/weights/best.pt'
)

parser.add_argument(
    '--source',
    required=True,
    help='Path to image, video, folder, or webcam index (example: 0)'
)

parser.add_argument(
    '--thresh',
    type=float,
    default=0.5,
    help='Minimum confidence threshold. Default = 0.5'
)

parser.add_argument(
    '--resolution',
    default=None,
    help='Optional display/output resolution in WxH format, example: "640x480"'
)

parser.add_argument(
    '--record',
    action='store_true',
    help='Enable recording of output video (only works for webcam or video)'
)

args = parser.parse_args()

# Parse arguments
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Load model
if not os.path.exists(model_path):
    print("ERROR: model not found.")
    sys.exit(1)

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
labels = model.names

# Determine source type
img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.avi','.mp4','.mov','.mkv']

if os.path.isdir(img_source):
    source_type = 'folder'

elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1].lower()
    source_type = 'image' if ext in img_ext else 'video'

elif img_source.isdigit():
    source_type = 'webcam'
    cam_index = int(img_source)

else:
    print("ERROR: Unsupported source")
    sys.exit(1)

# Parse resolution
resize = False
if user_res:
    resW, resH = map(int, user_res.split('x'))
    resize = True

# Prepare recording
if record:
    if source_type not in ['video', 'webcam']:
        print("Recording supported only for webcam or video.")
        sys.exit(1)
    recorder = cv2.VideoWriter(
        "demo_output.avi",
        cv2.VideoWriter_fourcc(*'MJPG'),
        30,
        (resW, resH)
    )

# Load image list or camera
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + "/*") if os.path.splitext(f)[1].lower() in img_ext]

elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)

elif source_type == 'webcam':
    cap = cv2.VideoCapture(cam_index)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

# YOLO Inference loop
img_count = 0
avg_fps = 0
fps_buffer = []

while True:

    start = time.perf_counter()

    # Load frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    else:  # Webcam or video
        ret, frame = cap.read()
        if not ret:
            break

    # Resize if needed
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run YOLO
    results = model(frame, verbose=False, device=device)
    detections = results[0].boxes

    # Draw detections
    for det in detections:
        conf = det.conf.item()
        if conf < min_thresh:
            continue
        x1, y1, x2, y2 = map(int, det.xyxy.squeeze().tolist())
        cls = int(det.cls.item())
        name = labels[cls]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # FPS calculation
    end = time.perf_counter()
    fps = 1 / (end - start)
    fps_buffer.append(fps)
    if len(fps_buffer) > 30:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)

    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # Display
    cv2.imshow("YOLO Detection", frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Cleanup
if source_type in ['video', 'webcam']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

print("Finished!")
