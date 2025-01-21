import streamlit as st
import requests
import torch
import cv2
import numpy as np
import tempfile
import os
import pathlib

pathlib.PosixPath = pathlib.WindowsPath

GITHUB_API_URL = "https://api.github.com/repos/o0omax/AppliedDeepLearning/releases/latest"
LOCAL_MODEL_PATH = "weights/best.pt"
USE_LOCAL_FOR_TESTING = True

@st.cache_data
def get_latest_model_url():
    r = requests.get(GITHUB_API_URL)
    r.raise_for_status()
    data = r.json()
    for asset in data.get("assets", []):
        if asset["name"] == "best.pt":
            return asset["browser_download_url"]
    raise ValueError("No best.pt in release.")

@st.cache_data
def download_model_cached(url: str) -> str:
    tdir = tempfile.mkdtemp()
    path = os.path.join(tdir, "best.pt")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path

@st.cache_resource
def load_model_cached(use_local: bool = USE_LOCAL_FOR_TESTING, local_path: str = LOCAL_MODEL_PATH):
    if use_local:
        if not os.path.exists(local_path):
            url = get_latest_model_url()
            model_path = download_model_cached(url)
        else:
            model_path = local_path
    else:
        url = get_latest_model_url()
        model_path = download_model_cached(url)
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
    model.conf = 0.1
    model.iou = 0.5
    return model

def draw_boxes(image: np.ndarray, results, conf_threshold: float) -> np.ndarray:
    out = image.copy()
    if len(results.xyxy) > 0:
        dets = results.xyxy[0].cpu().numpy()
        for *xyxy, conf, cls_id in dets:
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            name = results.names[int(cls_id)]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{name} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

def process_image(uploaded_file, model, conf_threshold: float):
    b = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(b, cv2.IMREAD_COLOR)
    r = model(img)
    out = draw_boxes(img, r, conf_threshold)
    d = r.xyxy[0].cpu().numpy() if len(r.xyxy) > 0 else None
    return out, d

def process_video(video_path: str, model, conf_threshold: float):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    c = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        c += 1
        r = model(frame)
        out = draw_boxes(frame, r, conf_threshold)
        stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f"Frame {c}", use_container_width=True)
    cap.release()

def process_webcam_live_cv2(model, conf_threshold: float, num_frames: int):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        r = model(frame)
        out = draw_boxes(frame, r, conf_threshold)
        stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
    cap.release()
    stframe.empty()
