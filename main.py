import pickle
import numpy as np
import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')

from flask import Flask, render_template, request
app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    img_path = 'static/sampimg.jpeg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction=model.predict([x])
    output=decode_predictions(prediction, top=5)[0]
    return render_template('index.html',prediction_text=output)

if __name__ =="__main__":
    app.run(debug=True)