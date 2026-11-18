"""Flask web app for pneumonia detection from chest X-ray images.

Routes:
    GET  /         Upload page.
    POST /predict  Multipart form: ``image`` (file) and ``models`` (JSON list of
                   model ids). Returns ``{"probability", "pneumonia", "per_model"}``.

Run:
    python app.py
"""

import json

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

import predictor

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    models = {model_id: name for model_id, (name, _) in predictor.AVAILABLE_MODELS.items()}
    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        model_ids = json.loads(request.form.get("models", "[]"))
        image = Image.open(image_file.stream)
        result = predictor.predict(image, model_ids)
    except UnidentifiedImageError:
        return jsonify({"error": "The file is not a valid image"}), 400
    except (ValueError, TypeError) as exc:  # bad JSON, no/unknown model
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
