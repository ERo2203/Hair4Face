import cv2
import numpy as np
try:
    import mediapipe as mp
except Exception:
    mp = None

mp_face_mesh = mp.solutions.face_mesh if mp is not None else None


def _to_px(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=float)


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_facial_metrics(image_path: str) -> dict:
    """
    Compute core facial metrics using MediaPipe FaceMesh.

    Returns a dict with:
      - face_width, face_length, interocular, nose_width (pixels)
      - thirds: {upper, middle, lower} normalized to face_length
      - keypoints: subset of pixel coords used
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mp_face_mesh is None:
        raise RuntimeError("MediaPipe not available in this environment.")

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as fm:
        res = fm.process(rgb)

    if not res.multi_face_landmarks:
        raise ValueError("No face detected")

    lm = res.multi_face_landmarks[0].landmark

    # Landmarks
    LEFT_EYE_IN = 133
    RIGHT_EYE_IN = 362
    # Primary alar landmarks
    NOSE_LEFT = 49
    NOSE_RIGHT = 279
    # Alternate alar/extreme wing candidates to improve robustness on some faces
    NOSE_LEFT_ALT = 98
    NOSE_RIGHT_ALT = 327
    NOSE_BASE = 2       # base of nose
    CHIN = 152          # chin bottom
    BROW_GLABELLA = 9   # glabella region between brows

    pts = {
        "left_eye_inner": _to_px(lm[LEFT_EYE_IN], w, h),
        "right_eye_inner": _to_px(lm[RIGHT_EYE_IN], w, h),
        "nose_left": _to_px(lm[NOSE_LEFT], w, h),
        "nose_right": _to_px(lm[NOSE_RIGHT], w, h),
        "nose_base": _to_px(lm[NOSE_BASE], w, h),
        "chin": _to_px(lm[CHIN], w, h),
        "brow_glabella": _to_px(lm[BROW_GLABELLA], w, h),
    }

    # Forehead estimation via extrapolation from mid-eye to chin
    mid_eye = (pts["left_eye_inner"] + pts["right_eye_inner"]) / 2.0
    direction = mid_eye - pts["chin"]
    FOREHEAD_FACTOR = 0.65
    forehead = mid_eye + direction * FOREHEAD_FACTOR
    forehead[0] = float(np.clip(forehead[0], 0, w - 1))
    forehead[1] = float(np.clip(forehead[1], 0, h - 1))

    face_width = _euclid(pts["left_eye_inner"], pts["right_eye_inner"]) + _euclid(pts["nose_left"], pts["nose_right"])  # proxy; width between cheeks not directly measured here
    # Prefer true width using lateral face landmarks when available
    LEFT_FACE = None
    RIGHT_FACE = None
    try:
        LEFT_FACE = _to_px(lm[234], w, h)
        RIGHT_FACE = _to_px(lm[454], w, h)
        face_width = _euclid(LEFT_FACE, RIGHT_FACE)
    except Exception:
        pass

    face_length = _euclid(forehead, pts["chin"])
    interocular = _euclid(pts["left_eye_inner"], pts["right_eye_inner"])

    # Nose width: choose widest pair among primary and alternate alar candidates (improves consistency)
    try:
        nose_left_alt = _to_px(lm[NOSE_LEFT_ALT], w, h)
        nose_right_alt = _to_px(lm[NOSE_RIGHT_ALT], w, h)
        nose_pairs = [
            (pts["nose_left"], pts["nose_right"]),
            (nose_left_alt, nose_right_alt),
            (pts["nose_left"], nose_right_alt),
            (nose_left_alt, pts["nose_right"]),
        ]
        nose_width = max(_euclid(a, b) for a, b in nose_pairs)
    except Exception:
        nose_width = _euclid(pts["nose_left"], pts["nose_right"])  # fallback

    # Correct facial thirds: EXTRAPOLATED forehead→brow, brow→nose base, nose base→chin
    upper_third = abs(pts["brow_glabella"][1] - forehead[1])
    middle_third = abs(pts["nose_base"][1] - pts["brow_glabella"][1])
    lower_third = abs(pts["chin"][1] - pts["nose_base"][1])

    thirds = {
        "upper": float(upper_third / face_length) if face_length > 0 else 0.0,
        "middle": float(middle_third / face_length) if face_length > 0 else 0.0,
        "lower": float(lower_third / face_length) if face_length > 0 else 0.0,
    }

    return {
        "face_width": face_width,
        "face_length": face_length,
        "interocular": interocular,
        "nose_width": nose_width,
        "thirds": thirds,
        "keypoints": {k: (float(v[0]), float(v[1])) for k, v in pts.items()},
        "overlay_points": {
            "forehead": (float(forehead[0]), float(forehead[1])),
            "brow_glabella": (float(pts["brow_glabella"][0]), float(pts["brow_glabella"][1])),
            "nose_base": (float(pts["nose_base"][0]), float(pts["nose_base"][1])),
            "chin": (float(pts["chin"][0]), float(pts["chin"][1])),
            "left_eye_inner": (float(pts["left_eye_inner"][0]), float(pts["left_eye_inner"][1])),
            "right_eye_inner": (float(pts["right_eye_inner"][0]), float(pts["right_eye_inner"][1])),
            "nose_left": (float(pts["nose_left"][0]), float(pts["nose_left"][1])),
            "nose_right": (float(pts["nose_right"][0]), float(pts["nose_right"][1])),
            "left_face": (float(LEFT_FACE[0]), float(LEFT_FACE[1])) if LEFT_FACE is not None else None,
            "right_face": (float(RIGHT_FACE[0]), float(RIGHT_FACE[1])) if RIGHT_FACE is not None else None,
            "image_size": (int(w), int(h)),
        },
    }


def classify_interocular(interocular: float, face_width: float) -> str:
    ratio = interocular / max(face_width, 1e-6)
    if ratio < 0.24:
        return "narrow"
    if ratio <= 0.30:
        return "average"
    return "wide"


def classify_nose_width(nose_width: float, face_width: float) -> str:
    ratio = nose_width / max(face_width, 1e-6)
    if ratio < 0.28:
        return "narrow"
    if ratio <= 0.35:
        return "average"
    return "wide"


def classify_face_shape(face_length: float, face_width: float, thirds: dict, gender: str) -> str:
    ratio = face_length / max(face_width, 1e-6)
    upper, middle, lower = thirds["upper"], thirds["middle"], thirds["lower"]

    # Basic heuristic; can be tuned per gender
    if 1.3 <= ratio <= 1.45 and all(0.28 <= t <= 0.38 for t in (upper, middle, lower)):
        return "oval"
    if ratio < 1.25:
        return "round"
    if ratio > 1.55:
        return "oblong"
    if abs(lower - upper) < 0.05 and 1.25 <= ratio <= 1.5:
        return "square"
    if gender.lower().startswith("w") and upper > 0.37:
        return "heart"
    return "oval"


def run_metrics_inference(image_path: str, gender: str) -> dict:
    metrics = compute_facial_metrics(image_path)
    face_width = metrics["face_width"]
    interocular_cat = classify_interocular(metrics["interocular"], face_width)
    nose_cat = classify_nose_width(metrics["nose_width"], face_width)
    face_shape = classify_face_shape(metrics["face_length"], face_width, metrics["thirds"], gender)

    return {
        "face_shape": face_shape,
        "interocular_category": interocular_cat,
        "nose_category": nose_cat,
        "thirds": metrics["thirds"],
        "overlay_points": metrics.get("overlay_points"),
    }





