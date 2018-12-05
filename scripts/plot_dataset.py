from preprocessing.dataset import SVHNDataset, ColorConverter, SVHNPlotter
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from keras import optimizers
import configparser as cp
from datetime import datetime as dt
import os
from shutil import copyfile
from plot.plot_autoencoder import plot_ae

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")

    batch_size = config["general"].getint("batch_size")
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    noise_ratio = config["general"].getfloat("noise_ratio")
    train_set = SVHNDataset.from_npy(config["general"].get("training_set"))
    dev_set = SVHNDataset.from_npy(config["general"].get("dev_set"))
    print(f"Training Set Color_Mode is {train_set.color_mode}")
    print(f"Dev Set Color_Mode is {train_set.color_mode}")
    if color_mode == "grayscale":
        converter = ColorConverter(color_mode)
        train_set = converter.transform(train_set)
        dev_set = converter.transform(dev_set)
    plotter = SVHNPlotter(output_dir=f"images/{train_set.name}")
    plotter.save_images(train_set, n=10)
    plotter.save_mosaic(train_set, row=10, col=10)

    trn_gen = train_set.generator(batch_size=100, flatten=False, noise=0.05)
    print(trn_gen[0][0].shape)
    print(trn_gen[0][1].shape)
    train_set_gen = SVHNDataset("trn_generator", images=trn_gen[0][0], labels=train_set.labels)
    plotter = SVHNPlotter(output_dir=f"images/{train_set_gen.name}")
    plotter.save_images(train_set_gen, n=10)
    plotter.save_mosaic(train_set_gen, row=10, col=10)

    train_set_gen = SVHNDataset("trn_generator_y", images=trn_gen[0][1], labels=train_set.labels)
    plotter = SVHNPlotter(output_dir=f"images/{train_set_gen.name}")
    plotter.save_images(train_set_gen, n=10)
    plotter.save_mosaic(train_set_gen, row=10, col=10)
