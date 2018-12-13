import numpy as np
from keras import metrics
from util.cnn_trainer import prepare_dataset
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from callbacks.save_encoder import SaveEncoder
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
    training_gen, validation_gen = prepare_dataset(config["general"].get("training_set"),
                                                   config["general"].get("dev_set"), color_mode=color_mode, ae=True,
                                                   batch_size=batch_size, noise_ratio=noise_ratio)

    tag = dt.now().strftime("%m_%d_%H%M%S") + f"_{color_mode}_ae_{ae_model}"
    log_dir = f"logs/{tag}"
    os.makedirs(log_dir, exist_ok=True)
    exp_dir = f"experiments/{tag}"
    os.makedirs(exp_dir, exist_ok=True)
    copyfile("config.ini", os.path.join(exp_dir, "config.ini"))
    print(f"experiment results will be written to {exp_dir}")

    if ae_model == "cnn":
        print(f"Training with CNN autoencoder")
        from models.ae.ae_cnn import svhn_ae

        autoencoder, encoder = svhn_ae(filter_size=config["ae_cnn"].getint("filter_size"),
                                       filter_number=config["ae_cnn"].getint("filter_number"),
                                       b_filter_number=config["ae_cnn"].getint("bottleneck_filter_number"),
                                       convs=config["ae_cnn"].getint("hidden_layers"),
                                       reg=config["ae_cnn"].getfloat("regularization"),
                                       color_mode=color_mode, batch_norm=config["ae_cnn"].getboolean("batch_norm"),
                                       droprate=config["ae_cnn"].getfloat("droprate"))

    else:
        print(f"Training with MLP autoencoder")
        from models.ae.ae_simple import autoencoder_model

        regularization = config["ae_mlp"].getfloat("regularization")
        autoencoder, encoder = autoencoder_model(color_mode=color_mode,
                                                 bottleneck_width=config["ae_mlp"].getint("bottleneck_width"),
                                                 expand_ratio=config["ae_mlp"].getfloat("expand_ratio"),
                                                 hidden_layers=config["ae_mlp"].getint("hidden_layers"),
                                                 reg=regularization)

    adam = optimizers.adam(lr=config["optimizer"].getfloat("lr"), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                           amsgrad=False)

    autoencoder.compile(optimizer=adam, loss='mse', metrics=[metrics.mape])
    with open(os.path.join(exp_dir, f"autoencoder.json"), "w+") as f:
        f.write(autoencoder.to_json())
    with open(os.path.join(exp_dir, f"encoder.json"), "w+") as f:
        f.write(encoder.to_json())

    callbacks = []
    callbacks.append(SaveEncoder(encoder, output_dir=exp_dir))
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
                    batch_size=batch_size,
                    write_graph=True,
                    write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
    autoencoder.summary()
    autoencoder.fit_generator(training_gen, validation_data=validation_gen, epochs=config["general"].getint("epoch"),
                              shuffle=True, callbacks=callbacks, verbose=config["training"].getint("verbosity"),
                              workers=4, use_multiprocessing=True)
    autoencoder.save_weights(os.path.join(exp_dir, f"autoencoder_final.h5"))
    plot_ae(config, tag=tag)
