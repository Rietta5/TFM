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

import torch
import piq


(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
Xtrain = Xtrain / 255
Xtest = Xtest / 255
Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)


Xtrain_big = np.zeros((Xtrain.shape[0],56,56,1))
Xtrain_big[:,14:14+28,14:14+28] = Xtrain[:,:,:]


dst_train = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain)).batch(128)

def var_pos(X, desp_h, desp_v):
  X_big = np.zeros((X.shape[0],56,56,1))
  X_big[:,14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
  return X_big


desps_h = range(-10,11)
desps_v = range(-10,11)
metric_LPIPS = piq.LPIPS(reduction="none", mean=[0., 0., 0.], std=[1., 1., 1.,])
metric_DISTS = piq.LPIPS(reduction="none", mean=[0., 0., 0.], std=[1., 1., 1.,])
metricas = {}
 
for desp_h in tqdm(desps_h):
    for desp_v in tqdm(desps_v):
        metricas[(desp_h, desp_v)] = {"LPIPS":[], "DISTS":[], "SSIM":[]}
        for img, label in dst_train:    
            img_desp = var_pos(img, desp_h = desp_h, desp_v = desp_v)
            img = var_pos(img, desp_h=0, desp_v=0)
            img = torch.from_numpy(img)
            img_desp = torch.from_numpy(img_desp)
            met_LPIPS = metric_LPIPS(img.permute(0,3,1,2), img_desp.permute(0,3,1,2))
            met_DISTS = metric_DISTS(img.permute(0,3,1,2), img_desp.permute(0,3,1,2))
            met_SSIM = piq.ssim(img.permute(0,3,1,2), img_desp.permute(0,3,1,2),reduction="none")
            metricas[(desp_h, desp_v)]["LPIPS"].extend(met_LPIPS)
            metricas[(desp_h, desp_v)]["DISTS"].extend(met_DISTS)
            metricas[(desp_h, desp_v)]["SSIM"].extend(met_SSIM)
            
            
df_metricas = pd.DataFrame.from_dict(metricas, orient="index").reset_index()
df_metricas.columns = ["desp_h","desp_v", "LPIPS", "DISTS", "SSIM"]

# saving the dataframe 
df_metricas.to_csv('IQA_metrics_TFM.csv') 