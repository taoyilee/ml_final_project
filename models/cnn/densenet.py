from preprocessing.dataset import SVHNDataset, ColorConverter
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from callbacks.save_encoder import SaveEncoder
from keras import optimizers
from keras.applications import DenseNet121
from keras.models import Model

import configparser as cp
from datetime import datetime as dt
import os
from shutil import copyfile
from plot.plot_autoencoder import plot_ae
from keras.layers import Input, Dense, GlobalAveragePooling2D


def dense_net(weights=None):
    bottleneck = DenseNet121(input_shape=(32, 32, 3), include_top=False,
                             weights=None)  # type: Model
    x = bottleneck.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', name="mlp_hidden0")(x)
    output = Dense(10, activation='sigmoid', name="output")(x)
    model = Model(inputs=bottleneck.input, outputs=output)
    return model
