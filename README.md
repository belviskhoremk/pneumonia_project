# 🫁 Pneumonia Detection from Chest X-Rays

A web app that predicts whether a chest X-ray shows **pneumonia**, using
convolutional neural networks trained with **transfer learning**. Upload an
image, choose one or more models, and get a probability. When several models
are selected, their predictions are averaged (a simple ensemble).

> ⚠️ Educational project. This is **not** a medical device and must not be used
> for diagnosis.

## Models

Both models start from networks pre-trained on ImageNet. Their convolutional
base is kept and the classification head is replaced:

```
MobileNet / MobileNetV2 base (ImageNet weights)
  → GlobalAveragePooling2D
  → Dense(512, ReLU)
  → Dropout(0.6)
  → Dense(1, sigmoid)        # P(pneumonia)
```

| Model id | Architecture | Weights file | Size |
|----------|--------------|--------------|------|
| `mobilenet` | MobileNet | `models/mobilenet.h5` | 45 MB |
| `mobilenet_v2` | MobileNetV2 | `models/mobilenet_v2.h5` | 35 MB |

- **Input**: 224×224 RGB, pixels scaled to [0, 1]
- **Loss**: binary cross-entropy. An output above **0.5** is reported as pneumonia.

## Project structure

```
.
├── app.py              # Flask routes: upload page and /predict API
├── predictor.py        # Model loading, preprocessing, ensemble prediction
├── models/             # Trained Keras models (.h5)
├── sample_images/      # Example X-rays (NORMAL-* and VIRUS-* pneumonia)
├── templates/index.html
└── static/             # script.js (upload + results UI), styles.css
```

## Getting started

**Requirements:** Python 3.9–3.11 (TensorFlow 2.15 does not support 3.12+).

```bash
git clone https://github.com/belviskhoremk/pneumonia_project.git
cd pneumonia_project

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

Open <http://127.0.0.1:5000>, pick an image from `sample_images/` and click **Predict**.

## API

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F image=@sample_images/VIRUS-9584831-0001.jpeg \
  -F 'models=["mobilenet", "mobilenet_v2"]'
```

```json
{
  "probability": 0.993,
  "pneumonia": true,
  "per_model": { "mobilenet": 0.99, "mobilenet_v2": 0.996 }
}
```

Errors (missing image, invalid file, no or unknown model) return HTTP 400
with `{"error": "..."}`.

## Results on the sample images

| | Predicted normal | Predicted pneumonia |
|---|---|---|
| **Normal (6)** | 3 | 3 |
| **Pneumonia (5)** | 0 | 5 |

The ensemble catches every pneumonia case but has a high **false-positive
rate** on normal X-rays. This is common for models trained on pneumonia
datasets, which usually contain about three times more pneumonia images than
normal ones. Possible improvements: class weighting, a tuned decision threshold,
and data augmentation.

## Tech stack

TensorFlow / Keras · Flask · Pillow · NumPy · vanilla JavaScript
