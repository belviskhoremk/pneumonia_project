"""Pneumonia classifier: load the trained CNNs and run predictions on X-rays.

Both models are MobileNet-family networks pre-trained on ImageNet and
fine-tuned on chest X-rays (transfer learning). The original classification
head was replaced by::

    GlobalAveragePooling2D -> Dense(512, relu) -> Dropout(0.6) -> Dense(1, sigmoid)

so each model outputs a single number: the probability of pneumonia.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODELS_DIR = Path(__file__).resolve().parent / "models"

# Model id (used by the web form) -> (display name, weights file).
AVAILABLE_MODELS = {
    "mobilenet": ("MobileNet", MODELS_DIR / "mobilenet.h5"),
    "mobilenet_v2": ("MobileNetV2", MODELS_DIR / "mobilenet_v2.h5"),
}

INPUT_SIZE = (224, 224)  # both networks expect 224x224 RGB images
THRESHOLD = 0.5          # probability above which we report pneumonia

# Load every model once at startup (loading takes a few seconds each).
_models = {model_id: load_model(path) for model_id, (_, path) in AVAILABLE_MODELS.items()}


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into the (1, 224, 224, 3) float batch the models expect.

    Pixels are scaled to [0, 1], the same rescaling used during training.
    """
    image = image.convert("RGB").resize(INPUT_SIZE)  # X-rays are often grayscale
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)  # add the batch dimension


def predict(image: Image.Image, model_ids: list[str]) -> dict:
    """Predict pneumonia with the selected models and average their outputs.

    Averaging several models (a simple *ensemble*) usually gives more stable
    predictions than any single model.

    Args:
        image: The chest X-ray.
        model_ids: Keys of ``AVAILABLE_MODELS`` to use (at least one).

    Returns:
        ``{"probability": float, "pneumonia": bool, "per_model": {id: float}}``

    Raises:
        ValueError: If no model or an unknown model is requested.
    """
    if not model_ids:
        raise ValueError("Select at least one model")
    unknown = set(model_ids) - _models.keys()
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")

    batch = preprocess(image)
    per_model = {
        model_id: float(_models[model_id].predict(batch, verbose=0)[0][0])
        for model_id in model_ids
    }
    probability = float(np.mean(list(per_model.values())))

    return {
        "probability": probability,
        "pneumonia": probability > THRESHOLD,
        "per_model": per_model,
    }
