from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import pandas as pd
from pickle import dump, load
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import cv2

from tensorflow.keras import layers
# tf.config.set_visible_devices([], device_type='GPU')
from tensorflow.keras.utils import get_file
from perceptnet.networks import PerceptNet




(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
Xtrain = Xtrain / 255
Xtest = Xtest / 255
Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)


Xtrain_big = np.zeros((Xtrain.shape[0],56,56,1))
Xtrain_big[:,14:14+28,14:14+28] = Xtrain[:,:,:]


dst_train = tf.data.Dataset.from_tensor_slices((Xtest, Ytest)).batch(256)

def var_pos(X, desp_h, desp_v):
  X_big = np.zeros((X.shape[0],56,56,1))
  X_big[:,14+desp_v:14+28+desp_v,14+desp_h:14+28+desp_h] = X[:,:,:]
  return X_big


model = PerceptNet(gdn_kernel_size=1, learnable_undersampling=False)
path_weights = get_file(fname='perceptnet_rgb.h5',
                        origin='https://github.com/Jorgvt/perceptnet/releases/download/Weights/final_model_rgb.h5')
model.build((None,56,56,3))
model.load_weights(path_weights)


desps_h = range(-10,11)
desps_v = range(-10,11)

metricas = {}
 
for desp_h in tqdm(desps_h):
    for desp_v in tqdm(desps_v):
        metricas[(desp_h, desp_v)] = {"MSE":[], "PerceptNet":[]}
        for img, label in dst_train:    
            img_desp = var_pos(img, desp_h = desp_h, desp_v = desp_v)
            img = var_pos(img, desp_h=0, desp_v=0)
            pred_img = model.predict(tf.image.grayscale_to_rgb(tf.convert_to_tensor(img)), verbose=0)
            pred_imgdesp = model.predict(tf.image.grayscale_to_rgb(tf.convert_to_tensor(img_desp)), verbose=0)
            met_perceptnet = (tf.reduce_sum((pred_img - pred_imgdesp)**2, axis = (1,2,3)))**(1/2)
            met_mse = tf.reduce_sum((img - img_desp)**2, axis = (1,2,3))
            metricas[(desp_h, desp_v)]["MSE"].extend(met_mse.numpy())
            metricas[(desp_h, desp_v)]["PerceptNet"].extend(met_perceptnet.numpy())
        
            
            
df_metricas = pd.DataFrame.from_dict(metricas, orient="index").reset_index()
df_metricas.columns = ["desp_h","desp_v", "MSE", "PerceptNet"]

# saving the dataframe 
df_metricas.to_csv('IQA_metrics_TFM_PerceptNet_MSE.csv') 