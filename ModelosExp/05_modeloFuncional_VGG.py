import os; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
  X_big = np.zeros((X.shape[0],56,56,1))
  X_big[:,14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
  return X_big.astype(np.float32), y.astype(np.int32)

def _fixup_shape(images, labels):
    images.set_shape((256,56,56,1))
    labels.set_shape((256,))
    return images, labels

dst_train = tf.data.Dataset.from_tensor_slices((Xtest, Ytest)).batch(256, drop_remainder=True)

var_pos_loquesea = partial(var_pos,desp_h = 0, desp_v = 0)

def tf_function(x,y):
  return tf.numpy_function(var_pos_loquesea, [x,y], (tf.float32, tf.int32))

dst_big = dst_train.map(tf_function).map(_fixup_shape)



Xtrain_big_VGG = dst_big.map(lambda x,y: (tf.keras.applications.vgg16.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None
),y) )


VGG16 = tf.keras.applications.vgg16.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(56,56,3),
)

for capa in VGG16.layers:
    capa.trainable = False

model_VGG16 = tf.keras.Sequential([
    VGG16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation = "softmax")
])

inputs = VGG16.input
GAPMP1 = layers.GlobalAveragePooling2D()(VGG16.layers[3].output)
GAPMP2 = layers.GlobalAveragePooling2D()(VGG16.layers[6].output)
GAPMP3 = layers.GlobalAveragePooling2D()(VGG16.layers[10].output)
GAPMP4 = layers.GlobalAveragePooling2D()(VGG16.layers[14].output)
GAPMP5 = layers.GlobalAveragePooling2D()(VGG16.layers[18].output)

GAPFinal = tf.concat([GAPMP1,GAPMP2,GAPMP3,GAPMP4,GAPMP5], -1)

outputs = layers.Dense(10, activation = "softmax")(GAPFinal)

ModeloVGGGAP = tf.keras.Model(inputs,outputs)

ModeloVGGGAP.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
ModeloVGGGAP.load_weights('./ModeloVGGGAP_15.h5')

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
        
        results_VGG = ModeloVGGGAP.evaluate(X_VGG)
        metricas_VGG[(desp_h, desp_v)] = results_VGG

with open(f"metricas_VGGGAP.pkl", "wb") as f:
    dump(metricas_VGG, f)


