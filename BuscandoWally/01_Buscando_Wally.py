from tqdm.auto import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# import plotly.express as px
from tensorflow.keras import layers
from pickle import dump, load
from pathlib import Path
import plotly.figure_factory as ff
from scipy.spatial import Delaunay
import plotly.graph_objects as go
import plotly.offline as pyo
import chart_studio.plotly as py
from scipy.interpolate import griddata
import os
from einops import rearrange
import json
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.init(project = "TFM_PruebasEqui", mode = "online")

X = []
Y = []
path = "./Images/Wally_desp/"
data = pd.read_csv("./Datasets/Dataset_Wally.csv")

for image,label in zip(data["X"],data["Y"]):
    img = cv2.imread(path + str(image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    X.append(img)

    label = np.array(eval(label))/500
    Y.append(label)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42)
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)
Xtest = np.array(Xtest)
Ytest = np.array(Ytest)

modelo_equiv = tf.keras.models.Sequential([
    layers.Conv2D(124, (7,7), activation = "relu", padding = "same" ,input_shape = Xtrain[0].shape),#56x56
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(2, activation  = "sigmoid")
])

modelo_equiv.compile(optimizer="adam", loss ="mse" , metrics=['mae']) #run_eagerly=True: No haga el grafo de operaciones -> Errores más entendibles
history = modelo_equiv.fit(Xtrain, Ytrain, batch_size=32, epochs=50, validation_data = (Xtest, Ytest),
                            callbacks = [WandbMetricsLogger(log_freq="epochs"),
                                         WandbModelCheckpoint(filepath=f"{wandb.run.dir}/clasificador", monitor = "val_loss", mode = "min", save_best_only = True, save_weights_only = True)])

wandb.finish()
    