import streamlit as st
import dicomsdl
import cv2
import tempfile
import os
import numpy as np
import pathlib

from fastai.vision.all import *

temp_posix = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='fastai')
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

RESIZE_TO = (1024, 1024)

# Stub out every custom function expected by the pickled DataBlock and Metrics
def label_func(path): return 0
def splitting_func(paths): return [], []
def get_items(image_dir_path): return []
def pfbeta_torch(preds, labels, beta=1): return 0.0
def pfbeta_torch_thresh(preds, labels): return 0.0
def optimize_preds(preds, labels=None, thresh=None, return_thresh=False, print_results=False): return preds

def dicom_file_to_ary(path):
    dcm_file = dicomsdl.open(str(path))
    data = dcm_file.pixelData()
    data = (data - data.min()) / (data.max() - data.min())
    if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = 1 - data
    data = cv2.resize(data, RESIZE_TO)
    return (data * 255).astype(np.uint8)

st.title("Breast Cancer DICOM Classifier")

if 'learner' not in st.session_state:
    try:
        st.session_state.learner = load_learner('export.pkl')
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    finally:
        pathlib.PosixPath = temp_posix

uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        st.info("Processing DICOM...")
        img_array = dicom_file_to_ary(tmp_path)
        
        # Convert 1-channel grayscale to 3-channel RGB
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        st.image(img_rgb, caption="Processed DICOM")
        
        # Write to a temporary PNG to perfectly mimic Kaggle's DataLoader
        png_path = tmp_path.replace(".dcm", ".png")
        cv2.imwrite(png_path, img_rgb)
        
        st.info("Running inference...")
        pred, pred_idx, probs = st.session_state.learner.predict(png_path)
        
        # Extract the specific probability for malignancy (class index 1)
        malignancy_probability = probs[1].item() * 100
        
        # Clinical interpretation mapping
        if pred_idx == 1 or malignancy_probability >= 50.0:
            result_text = "Positive for Malignancy (Suspicious Findings Detected)"
            st.error(f"**Diagnostic Assessment:** {result_text}")
        else:
            result_text = "Negative for Malignancy (No Suspicious Findings)"
            st.success(f"**Diagnostic Assessment:** {result_text}")
            
        st.write(f"**Model Confidence Score:** {malignancy_probability:.2f}%")
    except Exception as e:
        st.error(f"Error during processing: {e}")
    finally:
        # Cleanup both temporary files
        for p in [tmp_path, tmp_path.replace(".dcm", ".png")]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except PermissionError:
                    st.warning(f"Windows file lock prevented cleanup of {p}")