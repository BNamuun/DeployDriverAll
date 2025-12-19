import base64
import os
from typing import Any, Dict, List, Optional, Union
import logging
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")
app = FastAPI(title="Drowsiness ML Service")
logger = logging.getLogger(__name__)
# Allow browser calls during development (DriverApp runs on a different port).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model", "my_model.pt")

_model: Optional[YOLO] = None
_model_path: Optional[str] = None


def _resolve_model_path() -> str:
    """Resolve the YOLO model path.

    Priority:
      1) $MODEL_PATH (if set and exists)
      2) ./my_model/my_model.pt relative to this file
    """
    candidates: List[str] = []

    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.append(DEFAULT_MODEL_PATH)

    for p in candidates:
        ap = os.path.abspath(p)
        if os.path.exists(ap):
            return ap

    tried = ", ".join(os.path.abspath(p) for p in candidates)
    raise RuntimeError(f"MODEL_PATH not found. Tried: {tried}")


def _get_model() -> YOLO:
    global _model, _model_path
    if _model is None:
        _model_path = _resolve_model_path()
        _model = YOLO(_model_path)
    return _model


def _decode_image_base64(data: str) -> np.ndarray:
    """Decode a base64 image (raw base64 or data URL) into a BGR OpenCV image."""
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes")
    return img


def _name_for_class(names: Union[List[str], Dict[int, str]], cls_id: int) -> str:
    try:
        if isinstance(names, list):
            return names[cls_id]
        return names.get(cls_id, str(cls_id))
    except Exception:
        return str(cls_id)
    
    
def _derive_signals(class_names: List[str]) -> Dict[str, bool]:
    s = {n.lower().replace(" ", "_") for n in class_names}
    logger.info("normalized=%s", sorted(s))

    attentive_eye = any("attentive_eye" in n for n in s)
    drowsy_eye = any("drowsy_eye" in n for n in s)
    eye_closed = any("eyeclosed" in n or "eye_closed" in n for n in s)

    yawn = any("yawn" in n for n in s)
    asleep = any("asleep" in n for n in s)
    close = any(n == "close" or n.endswith("_close") or "close" in n for n in s)
    closed = any("closed" in n for n in s)

    signals = {
        "awake": attentive_eye,
        "yawn": yawn,
        "tired": drowsy_eye,
        "eye_closed": eye_closed,
        "asleep": asleep,
        "close": close,
        "closed": closed,
    }
    
    logger.info("derive_signals: signals=%s", signals)
    return signals

# def _derive_signals(class_names: List[str]) -> Dict[str, bool]:
#     s = {n.lower().replace(" ", "_") for n in class_names}

#     attentive_eye = any("Attentive eye" in n or "eye_closed" in n for n in s)
#     drowsy_eye = any("Drowsy eye" in n for n in s)
#     logger.info(f"tired={drowsy_eye}")
#     eye_closed = any("Eyeclosed" in n for n in s)

#     open_mouth = any("Open-Mouth" in n or n == "front" for n in s)
#     Yawn = any("Yawn" in n or n == "left" for n in s)
#     asleep = any("asleep" in n or n == "right" for n in s)
#     close = any("close" in n or n == "down" for n in s)
#     closed = any("closed" in n or n == "down" for n in s)
#     noYawn = any("noYawn" in n or n == "down" for n in s)
#     open = any("open" in n or n == "down" for n in s)
#     yawn = any("yawn" in n or n == "down" for n in s)
    
#     # looking_away = (face_left or face_right or face_down) and not face_front
#     # head_nod = face_down
#     signals = {
#         "awake": attentive_eye,
#         "yawn": yawn,
#         "tired": drowsy_eye,
#         "eye_closed": eye_closed,
#         "asleep": asleep,
#         "close": close,
#         "closed":closed
#     }  
#     logger.info("derive_signals: signals=%s", signals)
#     return signals


class InferBase64Request(BaseModel):
    image_base64: str = Field(..., description="Raw base64 bytes or a data URL")
    conf: float = Field(0.35, ge=0.0, le=1.0)
    imgsz: int = Field(640, ge=64, le=1920)


@app.get("/")
def root():
    """Root endpoint for Render health checks."""
    return {"status": "ok", "service": "Drowsiness ML Service"}


@app.get("/health")
def health():
    """Lightweight health check.

    Avoid loading the YOLO model here (model load can take multiple seconds and can
    cause proxy timeouts). /infer-base64 will lazy-load the model.
    """
    try:
        model_path = _resolve_model_path()
        return {
            "ok": True,
            "model_path": model_path,
            "model_loaded": _model is not None,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "model_path": os.path.abspath(DEFAULT_MODEL_PATH),
            "model_loaded": _model is not None,
        }


@app.post("/infer-base64")
def infer_base64(req: InferBase64Request):
    # console.log("dflsjfl")
    logger.info("qqqq:is working")
    model = _get_model()
    frame = _decode_image_base64(req.image_base64)

    results = model.predict(frame, conf=req.conf, imgsz=req.imgsz, verbose=False)
    r0 = results[0]
    logger.info("ro;", r0)

    detections: List[Dict[str, Any]] = []
    class_names: List[str] = []

    if r0.boxes is not None:
        for det in r0.boxes:
            conf = float(det.conf.item())
            cls_id = int(det.cls.item())
            xyxy = [float(x) for x in det.xyxy.squeeze().tolist()]
            cls_name = _name_for_class(model.names, cls_id)

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "conf": conf,
                    "xyxy": xyxy,
                }
            )
            class_names.append(cls_name)
    logger.info("class_names: %s", class_names)
    return {
        "detections": detections,
        "signals": _derive_signals(class_names),
    }
