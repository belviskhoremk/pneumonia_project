from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
def create_modified_model(model_path):
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    # Remove the last layer
    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # Add a new dense layer with a sigmoid activation function
    new_output = tf.keras.layers.Dense(2, activation='softmax')(model.output)
    # Create the modified model
    modified_model = tf.keras.Model(inputs=model.inputs, outputs=new_output)
    return modified_model

# Load pre-trained models
model1 = create_modified_model('model_mobilenet.h5')
# model1.summary()
model2 = create_modified_model('model_mobilenetv2.h5')
model3 = create_modified_model('model_xception.h5')