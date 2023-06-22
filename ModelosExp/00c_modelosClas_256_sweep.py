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

import wandb


(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)

#Xtrain_big = np.zeros((Xtrain.shape[0],256,256,1))
#Xtrain_big[:,114:114+28,114:114+28] = Xtrain[:,:,:]

config = {
    "batch_size": 256,
    "desp_h": 0,
    "desp_v": 0,
    "img_size": 256,
}

wandb.init(project="Sweep_VGG_ResNet_256",
           config=config)
config = wandb.config

def var_pos(X, y, desp_h, desp_v):
  X_big = np.zeros((X.shape[0],256,256,1))
  X_big[:,114+desp_v:114+28+desp_v,114+desp_h:114+28+desp_h] = X[:,:,:]
  return X_big.astype(np.float32), y.astype(np.int32)

def _fixup_shape(images, labels):
    images.set_shape((256,256,256,1))
    labels.set_shape((256,))
    return images, labels

dst_train = tf.data.Dataset.from_tensor_slices((Xtest, Ytest)).batch(config.batch_size, drop_remainder=True)

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
    input_shape=(256,256,3),
)

for capa in VGG16.layers:
    capa.trainable = False

    model_VGG16 = tf.keras.Sequential([
    VGG16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model_VGG16.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
model_VGG16.load_weights("model_VGG16.h5")

# ResNet50

Xtrain_big_RN = dst_big.map(lambda x,y: (tf.keras.applications.resnet50.preprocess_input(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )

ResNet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256,3)
)

for capa in ResNet50.layers:
    capa.trainable = False

model_ResNet50 = tf.keras.Sequential([
    ResNet50,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation = "softmax")
])

model_ResNet50.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
model_ResNet50.load_weights("model_ResNet50.h5")


# Metricas

# desps_h = range(-50,51)
# desps_v = range(-50,51)
metricas_VGG = {}
metricas_RN50 = {}
 
var_pos2 = partial(var_pos,desp_h = config.desp_h, desp_v = config.desp_v)

def tf_function2(x,y):
    return tf.numpy_function(var_pos2, [x,y], (tf.float32, tf.int32))

dst_big = dst_train.map(tf_function2).map(_fixup_shape)

X_VGG = dst_big.map(lambda x,y: (tf.keras.applications.vgg16.preprocess_input(
tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )

X_RN50 = dst_big.map(lambda x,y: (tf.keras.applications.resnet50.preprocess_input(
tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None),y) )

results_VGG = model_VGG16.evaluate(X_VGG, return_dict = True)
results_RN50 = model_ResNet50.evaluate(X_RN50, return_dict = True)

results_VGG = {f"{k}_VGG":v for k,v in results_VGG.items()}
results_RN50 = {f"{k}_RN50":v for k,v in results_RN50.items()}

wandb.log({
    **results_VGG,
    **results_RN50,
})

wandb.finish()

# metricas_VGG[(desp_h, desp_v)] = results_VGG
# metricas_RN50[(desp_h, desp_v)] = results_RN50

# with open(f"metricas_VGG3.pkl", "wb") as f:
#     dump(metricas_VGG, f)

# with open(f"metricas_RN503.pkl", "wb") as f:
#     dump(metricas_RN50, f)
