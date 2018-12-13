from preprocessing.dataset import SVHNDataset, ColorConverter
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from callbacks.save_encoder import SaveEncoder
from keras import optimizers
from keras.applications import DenseNet121
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
    print(f"Training Set Color_Mode is {train_set.color_mode}")
    print(f"Dev Set Color_Mode is {train_set.color_mode}")

    tag = dt.now().strftime("%m_%d_%H%M%S") + f"_{color_mode}_{ae_model}"
    log_dir = f"logs/{tag}"
    os.makedirs(log_dir, exist_ok=True)
    exp_dir = f"experiments/{tag}"
    os.makedirs(exp_dir, exist_ok=True)
    copyfile("config.ini", os.path.join(exp_dir, "config.ini"))
    print(f"experiment results will be written to {exp_dir}")

    print(f"Training with CNN")

    input_shape = train_set.images_shape
    regularization = config["ae_cnn"].getfloat("regularization")
    filter_size = (config["ae_cnn"].getint("filter_size"), config["ae_cnn"].getint("filter_size"))
    cnn_model = DenseNet121(input_shape, classes=10)
    gen = train_set.generator(batch_size=batch_size, ae=True, flatten=False, noise=noise_ratio)
    dev_gen = dev_set.generator(batch_size=batch_size, ae=True, flatten=False, noise=noise_ratio)

    print(f"input_shape is {input_shape}")
    adam = optimizers.adam(lr=config["optimizer"].getfloat("lr"), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                           amsgrad=False)

    cnn_model.compile(optimizer=adam, loss='mse')
    with open(os.path.join(exp_dir, f"autoencoder.json"), "w+") as f:
        f.write(cnn_model.to_json())

    callbacks = []
    callbacks.append(
        ModelCheckpoint(os.path.join(exp_dir, "full_weights.{epoch:02d}.h5"), monitor='val_loss', verbose=0,
                        save_best_only=False, save_weights_only=False, mode='auto', period=1))
    callbacks.append(CSVLogger(os.path.join(exp_dir, f"training.csv"), separator=',', append=False))
    callbacks.append(
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=config["training"].getint("patientce_reduce_lr"),
                          verbose=config["training"].getint("verbosity"), mode='min', min_delta=1e-6, cooldown=0,
                          min_lr=1e-12))
    if config["general"].getboolean("early_stop"):
        print("Early Stop enabled")
        callbacks.append(
            EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=5, verbose=1, mode='min',
                          baseline=None, restore_best_weights=False))
    else:
        print("Early Stop disabled")
    callbacks.append(
        TensorBoard(log_dir=log_dir, histogram_freq=0,
                    batch_size=32,
                    write_graph=True,
                    write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
    cnn_model.summary()
    cnn_model.fit_generator(gen, validation_data=dev_gen, epochs=config["general"].getint("epoch"), shuffle=True,
                            callbacks=callbacks, verbose=config["training"].getint("verbosity"), workers=4,
                            use_multiprocessing=True)
    cnn_model.save_weights(os.path.join(exp_dir, f"autoencoder_final.h5"))
    plot_ae(config, tag=tag)
