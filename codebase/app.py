import streamlit as st
import tempfile
import cv2

from codebase.metrics_utils import run_metrics_inference
from codebase.inference import predict_face_shape
from codebase.suggestions import fetch_suggestions


def fuse_face_shape(metrics_shape: str, model_probs: dict, gender: str) -> str:
    # Weighted: metrics 0.8, model 0.2
    weight_metrics = 0.8
    weight_model = 0.2

    # Normalize label space
    canonical = {
        "men": ["oblong", "oval", "round", "square"],
        "women": ["heart", "oblong", "oval", "round", "square"],
    }
    labels = canonical["women" if not gender.lower().startswith("m") else "men"]

    scores = {label: 0.0 for label in labels}
    if metrics_shape in scores:
        scores[metrics_shape] += weight_metrics
    for label, p in model_probs.items():
        if label in scores:
            scores[label] += weight_model * float(p)

    return max(scores.items(), key=lambda kv: kv[1])[0]


st.set_page_config(page_title="Hair4Face", page_icon="💇", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    :root {
      color-scheme: dark;
    }
    .stApp {
      background: #0f1116;
      color: #e6e6e6;
    }
    .stSidebar {
      background: #0b0d12 !important;
    }
    .stButton>button {
      background: #1f6feb; color: white; border-radius: 8px; border: 1px solid #265ebf;
    }
    .metric-card {background:#141824; padding:16px; border-radius:12px; border:1px solid #20263a}
    .tag {display:inline-block; padding:2px 8px; border-radius:999px; background:#1b2233; margin-right:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Hair4Face")
    gender = st.radio("Gender", ["Man", "Woman"], horizontal=True)
    st.caption("Choose your gender to select the appropriate model.")

st.title("Face Shape Analysis and Haircut Suggestions")
st.write("Capture a photo. We'll analyze facial thirds and shape, then suggest styles.")

img_file_buffer = st.camera_input("Camera")

if img_file_buffer is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_file_buffer.getvalue())
        image_path = tmp.name

    st.image(image_path, caption="Captured", use_container_width=True)

    # Metrics (MediaPipe)
    metrics_out = run_metrics_inference(image_path, gender)

    # Model
    model_out = predict_face_shape(image_path, gender)

    # Fusion
    fused_shape = fuse_face_shape(metrics_out["face_shape"], model_out.get("probs", {}), gender)

    # Compose final categories
    interocular_category = metrics_out["interocular_category"]
    nose_category = metrics_out["nose_category"]

    # Suggestions
    sugg = fetch_suggestions(gender, fused_shape, interocular_category, nose_category)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Face Overview")
        st.markdown(
            f"""
            <div class="metric-card">
            <div><span class="tag">Face shape</span> <b>{fused_shape.title()}</b></div>
            <div style="margin-top:8px"><span class="tag">Interocular</span> {interocular_category}</div>
            <div><span class="tag">Nose</span> {nose_category}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        thirds = metrics_out.get("thirds", {})
        st.markdown("**Facial Thirds (normalized)**")
        st.write({k: round(float(v), 2) for k, v in thirds.items()})

        # Annotated overlay
        overlay = metrics_out.get("overlay_points")
        if overlay:
            import numpy as np
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            draw = img.copy()

            def pt(name):
                p = overlay.get(name)
                return (int(p[0]), int(p[1])) if p else None

            # Horizontal guide lines at thirds
            forehead = pt("forehead")
            brow = pt("brow_glabella")
            noseb = pt("nose_base")
            chin = pt("chin")
            for y in [forehead[1], brow[1], noseb[1], chin[1]]:
                cv2.line(draw, (0, y), (w, y), (0, 255, 0), 2)

            # Face length (extrapolated forehead to chin)
            cv2.line(draw, forehead, chin, (255, 0, 0), 2)
            mid_fl = ((forehead[0]+chin[0])//2, (forehead[1]+chin[1])//2)
            cv2.putText(draw, "face length", (mid_fl[0]+6, mid_fl[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

            # Interocular
            le = pt("left_eye_inner")
            re = pt("right_eye_inner")
            cv2.line(draw, le, re, (255, 200, 0), 2)
            mid_eye = ((le[0]+re[0])//2, (le[1]+re[1])//2)
            cv2.putText(draw, "interocular", (mid_eye[0]+6, mid_eye[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)

            # Nose width (using measured endpoints we used for classification)
            nl = pt("nose_left")
            nr = pt("nose_right")
            cv2.line(draw, nl, nr, (0, 200, 255), 2)
            mid_nose = ((nl[0]+nr[0])//2, (nl[1]+nr[1])//2)
            cv2.putText(draw, "nose width", (mid_nose[0]+6, mid_nose[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

            # Face width if available
            lf = pt("left_face")
            rf = pt("right_face")
            if lf and rf:
                cv2.line(draw, lf, rf, (200, 0, 255), 2)
                mid_fw = ((lf[0]+rf[0])//2, (lf[1]+rf[1])//2)
                cv2.putText(draw, "face width", (mid_fw[0]+6, mid_fw[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,255), 1)

            # Also display numeric face length (px)
            def euclid(a, b):
                return float(((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5)
            face_length_px = euclid(forehead, chin)

            st.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), caption=f"Measurement overlay (face length: {face_length_px:.0f}px)", use_container_width=True)

    with col2:
        st.subheader("Hairstyle Suggestions")
        hair = sugg.get("hairstyle_suggestions")
        if hair:
            for s in str(hair).split(";"):
                st.markdown(f"- {s.strip()}")
        else:
            st.info("No specific hairstyle suggestions found.")

        if gender == "Man":
            beard = sugg.get("beard_suggestions")
            if beard:
                st.subheader("Beard Suggestions")
                for s in str(beard).split(";"):
                    st.markdown(f"- {s.strip()}")

    st.caption("Fusion uses MediaPipe metrics (0.8) + model prediction (0.2)")





