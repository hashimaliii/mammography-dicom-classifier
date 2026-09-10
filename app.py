"""
Breast Cancer DICOM Classifier — Streamlit App
==============================================
Model loaded from HuggingFace Hub.
Tab 1 — Demo Sample Cases  : browse labelled samples with live inference.
Tab 2 — Upload Your DICOM  : classify your own DICOM file.

Setup
-----
1. Push your export.pkl to HuggingFace:
       huggingface-cli login
       huggingface-cli upload <your-hf-username>/mammography-classifier export.pkl

2. Set HF_REPO_ID below (or via environment variable HF_REPO_ID).

3. Run:
       streamlit run app.py

Dependencies (requirements.txt):
       streamlit>=1.33
       huggingface_hub>=0.23
       fastai>=2.7
       dicomsdl
       opencv-python-headless
       pillow
       timm
"""

# ── 0. Cross-platform path fix (must run before any fastai import) ─────────
import platform, pathlib
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

# ── 1. Standard imports ────────────────────────────────────────────────────
import os, warnings, tempfile, math, time
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download, list_repo_files

# ── 2. Suppress noisy warnings ─────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning,  module="fastai")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# ── 3. fastai (import after path patch) ───────────────────────────────────
from fastai.vision.all import load_learner

# ── 4. Configuration ───────────────────────────────────────────────────────
# ⚠️  Replace this with your actual HuggingFace repo ID
HF_REPO_ID   = os.environ.get("HF_REPO_ID", "YOUR_HF_USERNAME/mammography-classifier")
HF_FILENAME  = "export.pkl"          # filename inside the repo
SAMPLES_DIR  = pathlib.Path("samples")  # relative to app.py
RESIZE_TO    = (1024, 1024)
CONFIDENCE_THRESHOLD = 50.0          # % above which → positive

# ── 5. Stub functions expected by the pickled DataBlock ───────────────────
def label_func(path):                                          return 0
def splitting_func(paths):                                     return [], []
def get_items(image_dir_path):                                 return []
def pfbeta_torch(preds, labels, beta=1):                       return 0.0
def pfbeta_torch_thresh(preds, labels):                        return 0.0
def optimize_preds(preds, labels=None, thresh=None,
                   return_thresh=False, print_results=False):  return preds

# ── 6. DICOM helpers ───────────────────────────────────────────────────────
def dicom_to_array(path: str | pathlib.Path) -> np.ndarray:
    """Load a DICOM → normalised uint8 numpy array [H, W]."""
    import dicomsdl
    dcm  = dicomsdl.open(str(path))
    data = dcm.pixelData().astype(np.float32)
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo)
    if dcm.getPixelDataInfo()["PhotometricInterpretation"] == "MONOCHROME1":
        data = 1.0 - data
    data = cv2.resize(data, RESIZE_TO, interpolation=cv2.INTER_LINEAR)
    return (data * 255).astype(np.uint8)


def dicom_to_pil(path: str | pathlib.Path, thumb_size: int = 256) -> Image.Image:
    """Return a PIL image thumbnail for display."""
    arr = dicom_to_array(path)
    img = Image.fromarray(arr).convert("RGB")
    img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
    return img


def run_inference(learner, dicom_path: str | pathlib.Path) -> dict:
    """
    Run the model on a DICOM file.
    Returns a dict with keys: label, confidence, raw_probs.
    """
    temp_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath          # fastai Windows compat

    arr     = dicom_to_array(dicom_path)
    img_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img_rgb)

    try:
        pred, pred_idx, probs = learner.predict(tmp_path)
        malignancy_pct = float(probs[1]) * 100.0
        return {
            "label"      : int(pred_idx),
            "confidence" : malignancy_pct,
            "raw_probs"  : [float(p) for p in probs],
        }
    finally:
        pathlib.PosixPath = temp_posix
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── 7. Model loader (cached across reruns) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model() -> object:
    """
    Download export.pkl from HuggingFace Hub and load via fastai.
    Result is cached for the lifetime of the Streamlit server process.
    """
    model_path = hf_hub_download(
        repo_id  = HF_REPO_ID,
        filename = HF_FILENAME,
    )
    learner = load_learner(model_path)
    learner.model.eval()
    return learner


# ── 8. Sample browser helpers ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_sample_files() -> dict[str, list[pathlib.Path]]:
    """Return {"negative": [...], "positive": [...]} from the samples/ folder."""
    result = {"negative": [], "positive": []}
    for label in result:
        folder = SAMPLES_DIR / label
        if folder.exists():
            result[label] = sorted(folder.glob("*.dcm"))
    return result


# ── 9. Shared UI components ───────────────────────────────────────────────
def render_result_card(confidence: float, label: int) -> None:
    """Render a styled result card."""
    is_positive = label == 1 or confidence >= CONFIDENCE_THRESHOLD

    if is_positive:
        st.markdown(
            f"""
            <div style="
                background: #fef2f2;
                border-left: 4px solid #dc2626;
                border-radius: 6px;
                padding: 14px 18px;
                margin-top: 12px;
            ">
                <div style="font-weight:600; color:#dc2626; font-size:15px;">
                    ⚠ Suspicious Findings Detected
                </div>
                <div style="color:#7f1d1d; margin-top:4px; font-size:13px;">
                    Model confidence: <strong>{confidence:.1f}%</strong> for malignancy
                </div>
                <div style="color:#991b1b; margin-top:6px; font-size:12px;">
                    This is a screening aid only. Clinical review required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="
                background: #f0fdf4;
                border-left: 4px solid #16a34a;
                border-radius: 6px;
                padding: 14px 18px;
                margin-top: 12px;
            ">
                <div style="font-weight:600; color:#16a34a; font-size:15px;">
                    ✓ No Suspicious Findings
                </div>
                <div style="color:#14532d; margin-top:4px; font-size:13px;">
                    Model confidence: <strong>{100 - confidence:.1f}%</strong> for benign
                </div>
                <div style="color:#166534; margin-top:6px; font-size:12px;">
                    This is a screening aid only. Clinical review required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_confidence_bar(confidence: float) -> None:
    """Horizontal bar showing malignancy probability."""
    red   = min(255, int(confidence * 2.55))
    green = min(255, int((100 - confidence) * 2.55))
    color = f"rgb({red},{green},60)"

    st.markdown(
        f"""
        <div style="margin-top:10px;">
            <div style="font-size:12px; color:#6b7280; margin-bottom:4px;">
                Malignancy probability
            </div>
            <div style="background:#e5e7eb; border-radius:4px; height:10px; width:100%;">
                <div style="
                    width:{confidence:.1f}%;
                    background:{color};
                    height:10px;
                    border-radius:4px;
                    transition: width 0.4s ease;
                "></div>
            </div>
            <div style="
                display:flex; justify-content:space-between;
                font-size:11px; color:#9ca3af; margin-top:3px;
            ">
                <span>0% (Benign)</span>
                <span>100% (Malignant)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── 10. Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Mammography Classifier",
    page_icon  = "🩻",
    layout     = "wide",
)

# ── 11. Global CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Remove Streamlit default top padding */
        .block-container { padding-top: 1.5rem; }

        /* Tab styling */
        button[data-baseweb="tab"] {
            font-size: 14px !important;
            font-weight: 500 !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            border: 2px dashed #d1d5db;
            border-radius: 8px;
            background: #f9fafb;
        }

        /* Divider between sample cards */
        hr { border-color: #e5e7eb; margin: 8px 0; }

        /* Pill badge */
        .badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
        }
        .badge-pos { background:#fee2e2; color:#dc2626; }
        .badge-neg { background:#dcfce7; color:#16a34a; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 12. Header ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:8px;">
        <span style="font-size:26px; font-weight:700; color:#111827;">
            🩻 Mammography Classifier
        </span>
        <span style="font-size:13px; color:#6b7280; margin-left:12px;">
            Deep learning screening aid · Not for clinical diagnosis
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 13. Load model (with visible spinner) ─────────────────────────────────
with st.spinner(f"Loading model from `{HF_REPO_ID}` …"):
    try:
        learner = load_model()
        st.success("Model ready.", icon="✅")
        time.sleep(0.6)
        st.empty()
    except Exception as exc:
        st.error(
            f"**Model failed to load.**\n\n"
            f"`{exc}`\n\n"
            f"Check that `HF_REPO_ID = '{HF_REPO_ID}'` is correct and "
            f"that `{HF_FILENAME}` exists in the repo.",
            icon="🚫",
        )
        st.stop()

# ── 14. Tabs ───────────────────────────────────────────────────────────────
tab_demo, tab_upload = st.tabs(["📋  Demo — Sample Cases", "⬆  Upload Your DICOM"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMO SAMPLE CASES
# ════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("Browse pre-loaded DICOM samples and see model predictions alongside ground-truth labels.")

    sample_files = get_sample_files()
    n_neg = len(sample_files["negative"])
    n_pos = len(sample_files["positive"])

    if n_neg + n_pos == 0:
        st.warning(
            "No sample files found. Expected folder structure:\n"
            "```\n"
            "samples/\n"
            "  negative/  *.dcm\n"
            "  positive/  *.dcm\n"
            "```",
            icon="📂",
        )
        st.stop()

    # ── Stats row
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total samples", n_neg + n_pos)
    col_b.metric("Negative (benign)",   n_neg, delta=None)
    col_c.metric("Positive (malignant)", n_pos, delta=None)
    st.divider()

    # ── Filter controls
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        show_label = st.selectbox(
            "Filter by ground truth",
            ["All", "Positive (cancer=1)", "Negative (cancer=0)"],
        )
    with c2:
        run_inference_flag = st.toggle("Run live inference on each sample", value=True)
    with c3:
        st.caption(
            "Live inference processes each DICOM through the model. "
            "Toggle off to browse thumbnails only."
        )

    # Build the display list
    if show_label == "Positive (cancer=1)":
        display_items = [("positive", p) for p in sample_files["positive"]]
    elif show_label == "Negative (cancer=0)":
        display_items = [("negative", p) for p in sample_files["negative"]]
    else:
        display_items = (
            [("positive", p) for p in sample_files["positive"]] +
            [("negative", p) for p in sample_files["negative"]]
        )

    if not display_items:
        st.info("No files match the current filter.")
    else:
        # ── Paginate: 6 cards per page
        CARDS_PER_PAGE = 6
        total_pages    = math.ceil(len(display_items) / CARDS_PER_PAGE)
        page           = st.number_input(
            f"Page (1–{total_pages})", min_value=1, max_value=total_pages,
            value=1, step=1, key="sample_page"
        )
        page_items = display_items[(page - 1) * CARDS_PER_PAGE : page * CARDS_PER_PAGE]

        cols = st.columns(3)
        for i, (gt_label, dcm_path) in enumerate(page_items):
            col = cols[i % 3]
            with col:
                # Thumbnail
                try:
                    thumb = dicom_to_pil(dcm_path, thumb_size=220)
                    col.image(thumb, use_container_width=True, clamp=True)
                except Exception as e:
                    col.error(f"Image load failed: {e}")

                # Ground truth badge
                badge_cls = "badge-pos" if gt_label == "positive" else "badge-neg"
                badge_txt = "cancer = 1" if gt_label == "positive" else "cancer = 0"
                col.markdown(
                    f'<span class="badge {badge_cls}">Ground truth: {badge_txt}</span>',
                    unsafe_allow_html=True,
                )
                col.caption(f"📄 {dcm_path.name}")

                # Live inference
                if run_inference_flag:
                    with col:
                        with st.spinner("Classifying…"):
                            try:
                                result = run_inference(learner, dcm_path)
                                render_confidence_bar(result["confidence"])
                                is_positive = (result["label"] == 1 or
                                               result["confidence"] >= CONFIDENCE_THRESHOLD)
                                pred_txt = "Malignant" if is_positive else "Benign"
                                pred_cls = "badge-pos" if is_positive else "badge-neg"
                                col.markdown(
                                    f'<span class="badge {pred_cls}" style="margin-top:4px;">Pred: {pred_txt} ({result["confidence"]:.1f}%)</span>',
                                    unsafe_allow_html=True,
                                )
                                # Agreement indicator
                                agreed = (gt_label == "positive") == is_positive
                                col.markdown(
                                    "✅ Correct" if agreed else "❌ Incorrect",
                                    help="Whether model prediction matches ground truth",
                                )
                            except Exception as e:
                                col.error(f"Inference failed: {e}")

                st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD YOUR DICOM
# ════════════════════════════════════════════════════════════════════════════
with tab_upload:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Upload a DICOM file")
        st.caption(
            "Supports standard mammography DICOM (.dcm). "
            "Files are processed locally and not stored."
        )
        uploaded_file = st.file_uploader(
            "Drop a .dcm file here or click to browse",
            type=["dcm"],
            key="user_upload",
            label_visibility="collapsed",
        )

        show_image = st.toggle("Show processed image", value=True, key="show_img_toggle")

        if uploaded_file is not None:
            # Write to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # ── Preview
                if show_image:
                    with st.spinner("Rendering DICOM…"):
                        try:
                            thumb = dicom_to_pil(tmp_path, thumb_size=400)
                            left.image(thumb, caption="Processed DICOM image", use_container_width=True)
                        except Exception as e:
                            left.warning(f"Could not render preview: {e}")

            finally:
                pass   # keep tmp_path alive for inference below

    with right:
        if uploaded_file is not None:
            st.subheader("Classification Result")
            st.markdown(f"**File:** `{uploaded_file.name}`  ·  **Size:** {len(uploaded_file.getvalue())/1024:.1f} KB")

            with st.spinner("Running model inference…"):
                try:
                    result = run_inference(learner, tmp_path)

                    # Confidence bar
                    render_confidence_bar(result["confidence"])

                    # Result card
                    render_result_card(result["confidence"], result["label"])

                    # Expandable detail
                    with st.expander("Raw model output"):
                        st.json({
                            "predicted_class"       : result["label"],
                            "malignancy_probability": f"{result['confidence']:.2f}%",
                            "benign_probability"    : f"{100 - result['confidence']:.2f}%",
                            "raw_probs"             : result["raw_probs"],
                            "threshold_used"        : f"{CONFIDENCE_THRESHOLD}%",
                        })

                    # Threshold explorer
                    with st.expander("Adjust decision threshold"):
                        custom_thresh = st.slider(
                            "Malignancy threshold (%)",
                            min_value=1, max_value=99,
                            value=int(CONFIDENCE_THRESHOLD),
                            step=1,
                            help="Lowering the threshold increases sensitivity (catches more cancers) "
                                 "but increases false positives.",
                        )
                        is_pos_custom = result["confidence"] >= custom_thresh
                        st.info(
                            f"At threshold **{custom_thresh}%**, this case is classified as: "
                            f"**{'Malignant ⚠' if is_pos_custom else 'Benign ✓'}**"
                        )

                except Exception as e:
                    st.error(f"Inference error: {e}")
                finally:
                    # Cleanup
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        else:
            # Empty state
            st.markdown(
                """
                <div style="
                    height: 320px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: #9ca3af;
                    border: 2px dashed #e5e7eb;
                    border-radius: 10px;
                    text-align: center;
                    padding: 24px;
                ">
                    <div style="font-size: 40px; margin-bottom: 12px;">🩻</div>
                    <div style="font-size: 15px; font-weight: 500; color: #6b7280;">
                        Upload a DICOM file to classify
                    </div>
                    <div style="font-size: 13px; margin-top: 8px;">
                        Results appear here after processing.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── 15. Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center; color:#9ca3af; font-size:11px; padding:4px 0 12px;">
        Model: <code>{HF_REPO_ID}</code> · 
        For research and educational use only · 
        Not validated for clinical screening decisions
    </div>
    """,
    unsafe_allow_html=True,
)