import numpy as np
import configparser as cp
import os
from util.fn import split_path
from preprocessing.dataset import SVHNDataset, SVHNPlotter, ColorConverter

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    original_train = os.path.join("dataset_split", "arrays", "train_32x32.npy")
    trainin_set = np.load(original_train)
    print(trainin_set.shape)

    training_ratio = config["general"].getfloat("train_ratio")
    training_number = int(len(trainin_set) * training_ratio / 100)
    print(f"Training samples: {training_number}")
    print(f"Validating samples: {len(trainin_set) - training_number}")

    index = list(range(len(trainin_set)))
    np.random.shuffle(index)

    trn_set = trainin_set[index[:training_number]]
    dev_set = trainin_set[index[training_number:]]
    print(f"Training samples: {len(trn_set)} {trn_set.shape}")
    print(f"Validating samples: {len(dev_set)} {dev_set.shape}")
    np.save(f"dataset_split/arrays/train_trn", trn_set)
    original_train = os.path.join("dataset_split", "arrays", "extra_32x32.npy")
    extra_set = np.load(original_train)
    trn_extra = np.concatenate((trn_set, extra_set), axis=0)
    print(f"train_extra has shape {trn_extra.shape}")
    np.save(f"dataset_split/arrays/train_trn_extra", trn_extra)
    np.save(f"dataset_split/arrays/train_dev", dev_set)
