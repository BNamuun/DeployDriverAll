---
title: Drowsiness Detection ML Service
emoji: 😴
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Drowsiness Detection ML Service

FastAPI-based machine learning service for real-time driver drowsiness detection using YOLO.

## Features
- Real-time eye state detection (attentive, drowsy, closed)
- Yawn detection
- Asleep state recognition
- Base64 image processing for easy integration
- GPU-accelerated inference

## API Endpoints

### `GET /`
Health check endpoint

### `GET /health`
Detailed health status including model path and device info

### `POST /infer-base64`
Main inference endpoint for drowsiness detection

**Request body:**
```json
{
  "image_base64": "base64_encoded_image_data",
  "conf": 0.35,
  "imgsz": 640
}
```

**Response:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "drowsy_eye",
      "conf": 0.85,
      "xyxy": [100, 150, 200, 250]
    }
  ],
  "signals": {
    "awake": false,
    "yawn": false,
    "tired": true,
    "eye_closed": false,
    "asleep": false,
    "close": false,
    "closed": false
  }
}
```

## Usage

Send POST requests to the `/infer-base64` endpoint with base64-encoded images to get real-time drowsiness detection results.

## Model

This service uses a custom-trained YOLO model for detecting various driver states and facial features related to drowsiness.
