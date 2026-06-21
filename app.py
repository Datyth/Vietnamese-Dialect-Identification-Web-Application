import io
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.features.mfcc import mfcc_mean_std
from src.utils.audio import load_audio, preprocess_waveform, TARGET_SAMPLE_RATE


MODEL_PATH = Path("outputs/models/svm_mfcc.pkl")


@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    # Training script saves a dict with keys including 'model' and 'label_order'.
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        label_order = obj.get("label_order")
        return model, label_order, obj
    # Otherwise assume it's a raw estimator
    label_order = getattr(obj, "classes_", None)
    return obj, label_order, {}


def predict_dialect(model, feature: np.ndarray, label_order=None) -> tuple[str, dict]:
    x = feature.reshape(1, -1)
    probs: dict = {}
    pred = model.predict(x)
    label = pred[0]
    # Map numeric index predictions back to label order if necessary
    try:
        if (isinstance(label, (int, np.integer)) or (isinstance(label, np.ndarray) and label.dtype.kind in "iu")) and label_order is not None:
            label = label_order[int(label)]
    except Exception:
        pass

    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(x)[0]
            classes = list(model.classes_)
            # Map classes to readable keys using label_order when classes are indices
            if label_order is not None and all(isinstance(c, (int, np.integer)) for c in classes):
                probs = {label_order[int(c)]: float(prob) for c, prob in zip(classes, p)}
            else:
                probs = {str(c): float(prob) for c, prob in zip(classes, p)}
        except Exception:
            probs = {}

    return label, probs


def main():
    st.title("Vietnamese Dialect Identification")
    st.write(
        "Upload a single-channel WAV file (any sample rate). The app resamples, trims/pads to 16s, extracts MFCC mean+std, and predicts Northern/Central/Southern dialect using the pretrained SVM MFCC model."
    )

    uploaded = st.file_uploader("Upload a .wav file", type=["wav","m4a"])

    if uploaded is None:
        st.info("Waiting for a WAV file to be uploaded.")
        return

    try:
        raw = uploaded.read()
        waveform, sample_rate = load_audio(raw)
    except Exception as exc:
        st.error(f"Could not read audio file: {exc}")
        return

    st.audio(raw)

    try:
        processed, stats = preprocess_waveform(waveform, sample_rate)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return

    st.write(
        {
            "original_sample_rate": int(sample_rate),
            "original_samples": int(waveform.size),
            "processed_sample_rate": int(stats.output_sample_rate),
            "processed_samples": int(stats.output_samples),
            "processed_duration_seconds": float(stats.output_duration_seconds),
        }
    )

    # Extract MFCC mean/std features
    try:
        feature = mfcc_mean_std(processed, sample_rate=TARGET_SAMPLE_RATE)
    except Exception as exc:
        st.error(f"Feature extraction failed: {exc}")
        return

    st.write(f"Feature vector length: {feature.size}")

    # Load model
    try:
        model, label_order, meta = load_model(MODEL_PATH)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    # Predict
    try:
        label, probs = predict_dialect(model, feature, label_order=label_order)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.header("Prediction")
    st.success(f"Predicted dialect: {label}")

    if probs:
        st.subheader("Probabilities")
        for k, v in probs.items():
            st.write(f"{k}: {v:.3f}")


if __name__ == "__main__":
    main()
