from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)
#
# def create_modified_model(model_path):
#     # Load the pre-trained model
#     model = tf.keras.models.load_model(model_path)
#     # Remove the last layer
#     model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
#     # Add a new dense layer with a sigmoid activation function
#     new_output = tf.keras.layers.Dense(1, activation='sigmoid')(model.output)
#     # Create the modified model
#     modified_model = tf.keras.Model(inputs=model.inputs, outputs=new_output)
#     return modified_model

# Load pre-trained models
model1 = load_model('MobileNet_model.h5')
# model1.summary()
model2 = load_model('MobileNetV2_model.h5')
# model3 = load_model('DenseNet169_model.h5')

models = {'model1': model1, 'model2': model2}

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert image to RGB if it is not
    image = image.resize((224, 224))  # Resize to the target size
    image = img_to_array(image)  # Convert image to array
    image = image / 255.0
    print(image.shape)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    # image = preprocess_input(image)  # Preprocess the image
    return image

# def preprocess_image(image):
#     img = np.array(image)
#     img = img / 255.0
#     img = img.reshape(1, 224, 224, 3)
#     return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image']
    selected_models = request.form['models']
    selected_models = json.loads(selected_models)
    print(selected_models)

    image = Image.open(io.BytesIO(image_file.read()))
    processed_image = preprocess_image(image)
    print(processed_image.shape)

    predictions = []
    for model_name in selected_models:
        model = models[model_name]
        prediction = model.predict(processed_image)
        print(prediction)
        # Assuming the pneumonia class is the second class
        pneumonia_probability = prediction[0][0]
        predictions.append(pneumonia_probability)  # Append probability of pneumonia

    avg_prediction = np.mean(predictions)
    result = {
        'probability': float(avg_prediction),
        'pneumonia': bool(avg_prediction > 0.5)  # Convert to Python bool
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
