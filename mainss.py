import pickle
import numpy as np
import keras
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from flask import Flask, render_template, request
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from keras.applications.resnet50 import ResNet50
model = VGG16(weights='imagenet')
model.save('models/model_VGG16.h5')