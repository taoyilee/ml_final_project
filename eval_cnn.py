from util.cnn_trainer import prepare_callbacks, prepare_dataset, prepare_model
import configparser as cp
from datetime import datetime as dt
import os
from shutil import copyfile
from preprocessing.dataset import SVHNDataset, ColorConverter
from keras.models import model_from_json, Model
import numpy as np

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config_train_cnn.ini")
    batch_size = config["general"].getint("batch_size")
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    noise_ratio = config["general"].getfloat("noise_ratio")

    print("*** PREPARING DATASET ***")
    test_set = SVHNDataset.from_npy(config["general"].get("test_set"))
    if color_mode == "grayscale":
        converter = ColorConverter(color_mode)
        test_set = converter.transform(test_set)

    test_gen = test_set.generator(batch_size=batch_size, ae=False, flatten=False, noise=None)

    tag = dt.now().strftime("%m_%d_%H%M%S") + f"_{color_mode}_{ae_model}"
    log_dir, exp_dir = f"logs/{tag}", f"experiments/{tag}"
    os.makedirs(log_dir, exist_ok=True), os.makedirs(exp_dir, exist_ok=True)

    with open(config["testing"].get("testing_model"), "r") as f:
        model = model_from_json(f.read())  # type: Model
    model.load_weights(config["testing"].get("testing_weights"))
    yhat = model.predict_generator(test_gen)
    yhat_argmax = np.argmax(yhat, axis=1) + 1
    y = test_set.labels
    correct = y == yhat_argmax
    print(f"y/yhat_argmax shape {y.shape} {yhat_argmax.shape}")
    misclass_rate = len(list(filter(lambda x: not x, correct))) / len(y)
    print(f"misclass_rate is {misclass_rate * 100:.2f}%")
