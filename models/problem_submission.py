from keras.utils import to_categorical
import numpy as np
from models.common import sample_and_split
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
from models.auc_callback import auc_callback
from models.dnn_libs import prepare_model
import random
import string
import configparser
import os

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_file = "config.ini"
    print(f"Reading {config_file}")
    config.read(config_file)
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    Xte = np.genfromtxt('data/X_test.txt', delimiter=None)
    raw_data = np.concatenate((X, Y), axis=1)
    regularization = 0
    trn_p = config["general"].getfloat("training_percentage")
    dev_p = 100 - trn_p

    tag = config["general"].get("run_tag")
    if tag is None:
        tag = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])

    print(f"Run tag is {tag}")
    work_dir = f"run_{tag}"
    exp_dir = os.path.join(config["general"].get("exp_dir"), f"{work_dir}")
    print(f"Experiment dir is {exp_dir}")
    os.makedirs(os.path.join(config["general"].get("log_dir"), f"{work_dir}"), exist_ok=True)
    os.makedirs(exp_dir, exist_ok=True)

    x_trn_raw, x_dev_raw, y_trn, y_dev, x_test_raw = sample_and_split(raw_data, Xte, train_percentage=trn_p,
                                                                      dev_percentage=dev_p, rescale=True,
                                                                      exp_dir=exp_dir)

    y_trn_encoded = to_categorical(y_trn)
    y_dev_encoded = to_categorical(y_dev)
    print(f"y_trn_encoded shape: {y_trn_encoded.shape}")
    print(f"y_trn_encoded[:10]: {y_trn_encoded[:10]}")
    print(f"Input dimension ={x_trn_raw.shape[1]}")
    model = prepare_model(x_trn_raw.shape[1], nodes_per_hidden=config["dnn"].getint("nodes_per_hidden"),
                          lr=config["dnn"].getfloat("lr"), hidden_layers=config["dnn"].getint("hidden_layers"),
                          regularization=0, exp_dir=exp_dir, weight_file=config["dnn"].get("initial_weights"))

    callbacks = []
    callbacks.append(auc_callback(x_trn_raw, y_trn_encoded, x_dev_raw, y_dev_encoded))
    callbacks.append(
        TensorBoard(log_dir=f"./logs/{work_dir}", histogram_freq=0, batch_size=32, write_graph=True,
                    write_grads=False,
                    write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='min',
                                       min_delta=1e-6, cooldown=0, min_lr=0))

    callbacks.append(
        EarlyStopping(monitor='last_dev_auc', min_delta=1e-7, patience=10, verbose=1, mode='max',
                      baseline=None, restore_best_weights=False))
    callbacks.append(CSVLogger(f"{exp_dir}/epoch_log.csv"))

    history = model.fit(x_trn_raw, y_trn_encoded, validation_data=(x_dev_raw, y_dev_encoded), epochs=100, verbose=2,
                        batch_size=1024, callbacks=callbacks)
    model.save(f"{exp_dir}/learner.h5")
    history.history["tag"] = tag
    history.history["hidden_layers"] = config["dnn"].getint("hidden_layers")
    history.history["nodes_per_hidden"] = config["dnn"].getint("nodes_per_hidden")
    for k, v in history.history.items():
        if isinstance(history.history[k], list):
            history.history[k] = history.history[k][-1]

    Yte = np.vstack((np.arange(Xte.shape[0]), model.predict(x_test_raw)[:, 1])).T
    np.savetxt(f"{exp_dir}/Y_submit.txt", Yte, ' %d, %.4f', header='ID,Prob1', comments='',
               delimiter=',')
