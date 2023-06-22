#import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from pickle import dump, load
import matplotlib.pyplot as plt
import cv2
import random
from IPython.display import clear_output
import tensorflow as tf
# import plotly.express as px
from tensorflow.keras import layers
from functools import partial
from PIL import Image
import re

import torch
import piq



(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)




def var_pos(X, y, desp_h, desp_v):
  X_big = np.zeros((X.shape[0],128,128,1))
  X_big[:,50+desp_v:50+28+desp_v,50+desp_h:50+28+desp_h] = X[:,:,:]
  return X_big.astype(np.float32), y.astype(np.int32)

def _fixup_shape(images, labels):
    images.set_shape((256,128,128,1))
    labels.set_shape((256,))
    return images, labels

dst_train = tf.data.Dataset.from_tensor_slices((Xtest, Ytest)).batch(256, drop_remainder=True)

var_pos_loquesea = partial(var_pos,desp_h = 0, desp_v = 0)

def tf_function(x,y):
  return tf.numpy_function(var_pos_loquesea, [x,y], (tf.float32, tf.int32))


dst_big = dst_train.map(tf_function).map(_fixup_shape)

#Xtrain_big = tf.keras.applications.vgg16.preprocess_input(
#    tf.image.grayscale_to_rgb(tf.convert_to_tensor(Xtrain_big)), data_format=None
#)

# VGG

Xtrain_big_VGG = dst_big.map(lambda x,y: (tf.keras.applications.vgg16.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None
),y) )

VGG16 = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(128,128,3),
)

for capa in VGG16.layers:
    capa.trainable = False

    model_VGG16 = tf.keras.Sequential([
    VGG16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model_VGG16.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
history = model_VGG16.fit(Xtrain_big_VGG, epochs = 10)


# ResNet50

Xtrain_big_RN = dst_big.map(lambda x,y: (tf.keras.applications.resnet50.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )

ResNet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(128,128,3)
)

for capa in ResNet50.layers:
    capa.trainable = False

model_ResNet50 = tf.keras.Sequential([
    ResNet50,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model_ResNet50.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")

history = model_ResNet50.fit(Xtrain_big_RN, epochs = 10)



# Metricas

desps_h = range(-10,11)
desps_v = range(-10,11)
metricas_VGG = {}
metricas_RN50 = {}
 
for desp_h in tqdm(desps_h):
    for desp_v in tqdm(desps_v): 
        var_pos2 = partial(var_pos,desp_h = desp_h, desp_v = desp_v)

        def tf_function2(x,y):
            return tf.numpy_function(var_pos2, [x,y], (tf.float32, tf.int32))

        dst_big = dst_train.map(tf_function2).map(_fixup_shape)

        X_VGG = dst_big.map(lambda x,y: (tf.keras.applications.vgg16.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )
        
        X_RN50 = dst_big.map(lambda x,y: (tf.keras.applications.resnet50.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )
        
        results_VGG = model_VGG16.evaluate(X_VGG)
        results_RN50 = model_ResNet50.evaluate(X_RN50)

        metricas_VGG[(desp_h, desp_v)] = results_VGG
        metricas_RN50[(desp_h, desp_v)] = results_RN50

with open(f"metricas_VGG2.pkl", "wb") as f:
    dump(metricas_VGG, f)

with open(f"metricas_RN502.pkl", "wb") as f:
    dump(metricas_RN50, f)
