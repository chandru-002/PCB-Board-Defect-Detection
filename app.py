import json
import os
from pathlib import Path
# pyright: reportMissingModuleSource=false

import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models") / "pcb_defect_model.h5"
CLASS_NAMES_PATH = Path("models") / "class_names.json"
MODEL_META_PATH = Path("models") / "model_meta.json"
REFERENCE_DATASET_PATH = Path("PCB_DATASET") / "images"
KNN_TOP_K = 1
ENABLE_MODEL_INFERENCE = os.environ.get("ENABLE_MODEL_INFERENCE", "0") == "1"
APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "5000"))
APP_DEBUG = os.environ.get("APP_DEBUG", "0") == "1"


app = Flask(__name__)


def _is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def _normalize_class_name(class_name):
    return class_name.strip().lower().replace(" ", "_").replace("-", "_")


def load_model_assets():
    model = None
    if ENABLE_MODEL_INFERENCE and MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"⚠️  Model load failed, using reference matcher only: {e}")
    elif not ENABLE_MODEL_INFERENCE:
        print("ℹ️  Model inference disabled; using reference matcher for predictions.")
    else:
        print(
            f"⚠️  Model file not found: {MODEL_PATH} (using reference matcher only)")

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = [
            "missing_hole",
            "mouse_bite",
            "open_circuit",
            "short",
            "spur",
            "spurious_copper",
        ]

    if MODEL_META_PATH.exists():
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        img_size = int(meta.get("img_size", 128))
    else:
        img_size = 128

    return model, class_names, img_size


def pretty_label(class_name):
    # Convert snake_case to title words and append "Defect".
    title = class_name.replace("_", " ").strip().title()
    return f"{title} Defect"


def preprocess_image(file_stream, img_size):
    image = Image.open(file_stream).convert("RGB")
    image = image.resize((img_size, img_size))
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def preprocess_image_path(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size))
    image_array = np.asarray(image, dtype="float32") / 255.0
    return np.expand_dims(image_array, axis=0)


def image_batch_to_embedding(image_batch):
    # Lightweight descriptor for fast startup and robust nearest-neighbor matching.
    img = image_batch[0]
    gray = np.mean(img, axis=2)
    small = Image.fromarray((gray * 255).astype(np.uint8)).resize((32, 32))
    vec = np.asarray(small, dtype=np.float32).flatten() / 255.0
    vec = vec - np.mean(vec)
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def build_reference_index(reference_root, img_size):
    embeddings = []
    labels = []

    if not reference_root.exists():
        return np.empty((0, 1024), dtype=np.float32), []

    for class_dir in sorted(reference_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = _normalize_class_name(class_dir.name)
        for file in class_dir.iterdir():
            if not file.is_file() or not _is_image_file(file.name):
                continue

            try:
                batch = preprocess_image_path(file, img_size)
                emb = image_batch_to_embedding(batch)
                embeddings.append(emb)
                labels.append(class_name)
            except Exception:
                continue

    if not embeddings:
        return np.empty((0, 1024), dtype=np.float32), []

    return np.vstack(embeddings).astype(np.float32), labels


def predict_with_reference_knn(image_batch, ref_embeddings, ref_labels, top_k=KNN_TOP_K):
    if len(ref_labels) == 0 or ref_embeddings.size == 0:
        return None, 0.0

    query = image_batch_to_embedding(image_batch)
    sims = ref_embeddings @ query

    # k=1 nearest neighbor gives class-specific matching and avoids
    # class-vote bias where one class can dominate top-k neighbors.
    best_idx = int(np.argmax(sims))
    best_label = ref_labels[best_idx]
    best_sim = float(sims[best_idx])
    confidence = max(0.0, min(100.0, ((best_sim + 1.0) / 2.0) * 100.0))
    return best_label, confidence


model, class_names, img_size = load_model_assets()
ref_embeddings, ref_labels = build_reference_index(
    REFERENCE_DATASET_PATH, img_size)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    defect_name = None
    error_text = None

    if request.method == "POST":
        if "image" not in request.files:
            error_text = "Please select an image file."
            return render_template(
                "index.html",
                prediction_text=prediction_text,
                defect_name=defect_name,
                error_text=error_text,
            )

        file = request.files["image"]
        if not file or file.filename == "":
            error_text = "Please select an image file."
            return render_template(
                "index.html",
                prediction_text=prediction_text,
                defect_name=defect_name,
                error_text=error_text,
            )

        try:
            image_batch = preprocess_image(file.stream, img_size)
            pred_class = None
            model_conf = 0.0

            if model is not None:
                probabilities = model.predict(image_batch, verbose=0)[0]
                pred_idx = int(np.argmax(probabilities))
                pred_class = class_names[pred_idx]
                model_conf = float(probabilities[pred_idx]) * 100.0

            # Use feature similarity against labeled reference images to improve
            # class reliability on this 6-defect PCB dataset.
            knn_class, knn_conf = predict_with_reference_knn(
                image_batch,
                ref_embeddings,
                ref_labels,
            )

            if knn_class is not None:
                pred_class = knn_class
                pred_conf = int(round(knn_conf))
            elif pred_class is not None:
                pred_conf = int(round(model_conf))
            else:
                raise RuntimeError(
                    "Prediction failed: no model and no reference matches available.")

            defect_name = pretty_label(pred_class)

            prediction_text = (
                f"Input Image: {file.filename}\n"
                f"Prediction: {defect_name}\n"
                f"Confidence: {pred_conf}%"
            )
        except Exception as exc:
            error_text = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        defect_name=defect_name,
        error_text=error_text,
    )


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
