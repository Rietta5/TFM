from tqdm.auto import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# import plotly.express as px
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from pickle import dump, load
from pathlib import Path

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


config = {
    "big_shape": (56,56),
    "kernel_size": 7,
    "filters":256,
    "epochs": 140,
    "pooling" : 2
}

wandb.init(project = "TFM_PruebasInv_Pooling",config = config, mode = "online")
config = wandb.config


# Carga de datos
(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
Xtrain = Xtrain / 255
Xtest = Xtest / 255
Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)

Xtrain_big = np.zeros((Xtrain.shape[0],*config.big_shape,1))
Xtrain_big[:,14:14+28,14:14+28] = Xtrain[:,:,:]

Xtest_big = np.zeros((Xtest.shape[0],*config.big_shape,1))
Xtest_big[:,14:14+28,14:14+28] = Xtest[:,:,:]

dst_train = tf.data.Dataset.from_tensor_slices((Xtrain_big, Ytrain))
dst_test = tf.data.Dataset.from_tensor_slices((Xtest_big, Ytest))

clasificador = tf.keras.models.Sequential([
    layers.Conv2D(config.filters, config.kernel_size, activation = "relu", padding = "same" ,input_shape = Xtrain_big[0].shape,use_bias=False),# kernel_regularizer = regularizers.L1(0.01) #28x28
    layers.MaxPooling2D(config.pooling),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation  = "softmax")
])

clasificador.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = clasificador.fit(Xtrain_big, Ytrain, batch_size=128, epochs=config.epochs, validation_data = (Xtest_big, Ytest), 
                           callbacks = [WandbMetricsLogger(log_freq="epochs"),
                                         WandbModelCheckpoint(filepath="clasificador", monitor = "val_accuracy", mode = "max", save_best_only = True, save_weights_only = True)])




def var_pos(X, desp_h, desp_v):
  X_big = np.zeros(shape=(*config.big_shape,1))
  X_big[14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
  return X_big

# def var_pos(X, desp_h, desp_v):
#   X_big = np.zeros((X.shape[0],*config.big_shape,1))
#   X_big[:,14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
#   X_pos = X_big[:,14:14+28,14:14+28]
#   return X_big

def generador():
    for x,y in zip(Xtrain,Ytrain):
        x_mov = var_pos(x,desp_h, desp_v)
        yield x_mov,y

def generador_test():
    for x,y in zip(Xtest,Ytest):
        x_mov = var_pos(x,desp_h, desp_v)
        yield x_mov,y


def out_mapas(model,X):
  capas = model.layers
  salida = [X]

  for capa in capas:
    salida_capa = capa(salida[-1]).numpy()
    salida.append(salida_capa)
  
  return salida


desps_h = range(-10,11)
desps_v = range(-10,11)
metricas = {}
mapas_caracteristicas = {}
 
for desp_h in tqdm(desps_h):
    for desp_v in tqdm(desps_v):
        
        dst_train_desp = tf.data.Dataset.from_generator(generador,
                                           output_signature=(tf.TensorSpec(shape=(*config.big_shape,1), dtype=tf.float32),
                                                             tf.TensorSpec(shape=(), dtype=tf.float32)))
        dst_test_desp = tf.data.Dataset.from_generator(generador_test,
                                           output_signature=(tf.TensorSpec(shape=(*config.big_shape,1), dtype=tf.float32),
                                                             tf.TensorSpec(shape=(), dtype=tf.float32)))

        met_train = clasificador.evaluate(dst_train_desp.batch(128), verbose=0, return_dict=True)
        met_test = clasificador.evaluate(dst_test_desp.batch(128), verbose=0, return_dict=True)
        metricas[(desp_h, desp_v)] = {"Train":met_train, "Test":met_test}

df_metricas = pd.DataFrame.from_dict(metricas, orient="index").reset_index()
df_metricas.columns = ["desp_h","desp_v", "Train", "Test"]

TrainLoss = []
TrainAccuracy = []
TestLoss = []
TestAccuracy = []

for key in metricas.keys():
    TrainLoss.append(metricas[key]["Train"]["loss"])
    TrainAccuracy.append(metricas[key]["Train"]["accuracy"])
    TestLoss.append(metricas[key]["Test"]["loss"])
    TestAccuracy.append(metricas[key]["Test"]["accuracy"])

df_metricas["TrainLoss"] = TrainLoss
df_metricas["TrainAccuracy"] = TrainAccuracy
df_metricas["TestLoss"] = TestLoss
df_metricas["TestAccuracy"] = TestAccuracy

df_metricas = df_metricas.drop("Train", axis = 1)
df_metricas = df_metricas.drop("Test", axis = 1)

wandb.log({"Metricas": wandb.Table(dataframe = df_metricas)})
wandb.finish()
    