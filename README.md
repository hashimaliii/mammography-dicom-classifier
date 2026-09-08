# Mammography DICOM Classifier

A lightweight local deployment of a deep learning model for screening mammography analysis, built using PyTorch, fastai, and Streamlit.

## Overview
This application processes raw 16-bit DICOM mammography files, normalizes them for neural network inference, and passes them through a `tf_efficientnetv2_s` binary classification model to evaluate potential malignancy markers.

## Features
- **DICOM Parsing**: Utilizes `dicomsdl` and `pydicom` to handle raw medical imaging data formats.
- **Streamlit Interface**: Provides an interactive local UI for rapid file uploads and real-time inference reporting.
- **Clinical Mapping**: Translates raw machine output probabilities into standardized radiological assessment phrasing.

## Requirements
- Python 3.10
- PyTorch
- fastai (v2.7+)
- timm (v0.6.12)
- Streamlit
- opencv-python
- dicomsdl

## Installation & Local Execution
1. Clone the repository to your local machine.
2. Install the required pinned dependencies to ensure namespace compatibility with the exported model architecture:
   ```bash
   pip install timm==0.6.12 fastai streamlit opencv-python pydicom dicomsdl