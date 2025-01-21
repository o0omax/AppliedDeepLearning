import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import requests
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

############################
# Einstellungen & Constants
############################
st.set_page_config(page_title="Flugzeugschaden-Erkennung (YOLOv5)", layout="centered")

GITHUB_API_URL = "https://api.github.com/repos/o0omax/AppliedDeepLearning/releases/latest"
USE_LOCAL_FOR_TESTING = True
LOCAL_MODEL_PATH = "weights/best.pt"  # Specify your local model file path

############################
# Utilities / Hilfsfunktionen
############################

def get_latest_model_url(api_url: str = GITHUB_API_URL) -> str:
    """
    Fetch the URL of the latest model file from GitHub Releases.
    """
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        release_data = response.json()

        # Search for the asset named "best.pt"
        for asset in release_data.get("assets", []):
            if asset["name"] == "best.pt":
                return asset["browser_download_url"]

        raise ValueError("Model file 'best.pt' not found in the latest release.")
    except Exception as e:
        raise RuntimeError(f"Error fetching model URL from GitHub: {e}")

def download_model(url: str) -> str:
    """
    Downloads the YOLOv5 model file from the given URL and saves it to a temporary location.
    """
    with st.spinner("Downloading YOLOv5 model from GitHub..."):
        try:
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "best.pt")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                f.write(response.content)

            st.success(f"Model downloaded successfully: {model_path}")
            return model_path
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            raise

def load_model(use_local: bool = False, local_path: str = LOCAL_MODEL_PATH, api_url: str = GITHUB_API_URL):
    """
    Loads the YOLOv5 model from either a local path or the latest GitHub release.
    """
    if use_local:
        if not os.path.exists(local_path):
            st.error(f"Local file not found: {local_path}")
            raise FileNotFoundError(f"Local file '{local_path}' does not exist.")
        st.warning(f"Using local model for testing: {local_path}")
        model_path = local_path
    else:
        model_url = get_latest_model_url(api_url)
        model_path = download_model(model_url)

    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local', force_reload=True)
    model.conf = 0.1
    model.iou = 0.5
    return model

def draw_boxes(image: np.ndarray, results, conf_threshold: float = 0.1) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image based on YOLOv5 results.
    """
    annotated_image = image.copy()

    if len(results.xyxy) > 0:
        detections = results.xyxy[0].cpu().numpy()
        for *xyxy, conf, cls_id in detections:
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            class_idx = int(cls_id)
            class_name = results.names[class_idx]
            color = (0, 255, 0)
            thickness = 2

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
    return annotated_image

def process_image(uploaded_file, model):
    """
    Processes an uploaded image file and returns the annotated image.
    """
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(image)
    annotated = draw_boxes(image, results)
    return annotated

############################
# Streamlit-Hauptfunktion
############################

def main():
    st.title("Flugzeugschaden-Erkennung (YOLOv5)")

    try:
        model = load_model(USE_LOCAL_FOR_TESTING)
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return

    st.sidebar.header("Input Options")
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ("Upload Image", "Upload Video", "Webcam (Photo)", "Webcam (Live)")
    )

    st.write("""
    **Detected Classes**:
    - "crack"
    - "dent"
    """)

    if input_type == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.info("Processing image...")
            with st.spinner("Running inference..."):
                annotated_img = process_image(uploaded_image, model)
            st.success("Processing complete!")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Result")
            result_name = "annotated_image.jpg"
            cv2.imwrite(result_name, annotated_img)
            with open(result_name, "rb") as file:
                st.download_button("Download Result", data=file, file_name=result_name, mime="image/jpg")

if __name__ == "__main__":
    main()
