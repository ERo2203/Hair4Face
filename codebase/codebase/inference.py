import numpy as np


_MEN_MODEL_PATH = "/Users/rohanrao/Desktop/hair4face/codebase/codebase/men_faceshape_model.h5"
_WOMEN_MODEL_PATH = "/Users/rohanrao/Desktop/hair4face/codebase/codebase/women_faceshape_model.h5"


def _preprocess(img_path: str, target_size=(128, 128)):
    try:
        from tensorflow.keras.preprocessing import image
    except Exception as e:
        raise RuntimeError("TensorFlow not available. Please install TensorFlow to use the CNN model.") from e
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_face_shape(img_path: str, gender: str) -> dict:
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        # Fallback: model not available
        return {"face_shape": None, "probs": {}}

    if gender.lower().startswith("m"):
        model = load_model(_MEN_MODEL_PATH)
        class_labels = ["oblong", "oval", "round", "square"]
    else:
        model = load_model(_WOMEN_MODEL_PATH)
        class_labels = ["Heart", "Oblong", "Oval", "Round", "Square"]

    x = _preprocess(img_path)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "face_shape": class_labels[idx].lower(),
        "probs": {class_labels[i].lower(): float(p) for i, p in enumerate(probs)},
    }





