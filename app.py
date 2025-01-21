import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from model_utils import (
    load_model_cached,
    process_image,
    process_video,
    process_webcam_live_cv2
)

def main():
    st.set_page_config(page_title="Flugzeugschaden-Erkennung (YOLOv5)", layout="centered")
    st.title("Flugzeugschaden-Erkennung (YOLOv5)")

    if "live_detection" not in st.session_state:
        st.session_state.live_detection = False

    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.1, 0.01)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ("Upload Image", "Upload Video", "Webcam (Photo)", "Webcam (Live with OpenCV)")
    )

    st.write("Detected Classes: crack, dent")

    try:
        model = load_model_cached()
        model.conf = conf_threshold
        model.iou = iou_threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    if input_type == "Upload Image":
        img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if img_file:
            with st.spinner("Inference..."):
                annotated, dets = process_image(img_file, model, conf_threshold)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
            if dets is not None and len(dets) > 0:
                st.write(f"Detections: {len(dets)}")
            out_name = "annotated_image.jpg"
            cv2.imwrite(out_name, annotated)
            with open(out_name, "rb") as f:
                st.download_button("Download Result", data=f, file_name=out_name, mime="image/jpg")

    elif input_type == "Upload Video":
        vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if vid_file:
            with st.spinner("Inference on video..."):
                tpath = os.path.join(tempfile.mkdtemp(), vid_file.name)
                with open(tpath, "wb") as out_file:
                    out_file.write(vid_file.read())
                process_video(tpath, model, conf_threshold)
            st.success("Done")

    elif input_type == "Webcam (Photo)":
        shot = st.camera_input("Take a photo")
        if shot:
            with st.spinner("Inference..."):
                annotated, dets = process_image(shot, model, conf_threshold)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
            if dets is not None and len(dets) > 0:
                st.write(f"Detections: {len(dets)}")
            out_name = "webcam_photo.jpg"
            cv2.imwrite(out_name, annotated)
            with open(out_name, "rb") as f:
                st.download_button("Download Result", data=f, file_name=out_name, mime="image/jpg")

    elif input_type == "Webcam (Live with OpenCV)":
        num_frames = st.slider("Frames to process", 1, 500, 200)
        if not st.session_state.live_detection:
            if st.button("Start detection"):
                st.session_state.live_detection = True

        if st.session_state.live_detection:
            if st.button("Stop detection"):
                st.session_state.live_detection = False
            else:
                process_webcam_live_cv2(model, conf_threshold, num_frames)

if __name__ == "__main__":
    main()
