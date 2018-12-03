from preprocessing.dataset import SVNHDataset
import os
import configparser as cp
import pandas as pd
import numpy as np
import PIL.Image


def dataset_image(config):
    if not os.path.exists("dataset_split/all.csv"):
        os.makedirs("dataset_split", exist_ok=True)
        train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
        file_names = train_set.save_for_viewing("dataset_split/images/training")
        df = pd.DataFrame()
        df["labels"] = train_set.labels.flatten()
        df["file_names"] = file_names
        df.to_csv("dataset_split/all.csv", index=False)


def dataset_split(config):
    df = pd.read_csv("dataset_split/all.csv")
    if not os.path.exists(config["general"].get("training_set")) or not os.path.exists(
            config["general"].get("dev_set")):
        df_shuffled = df.sample(frac=1)
        print(df_shuffled[:10])
        training_ratio = config["general"].getfloat("train_ratio")
        training_number = int(len(df) * training_ratio / 100)
        print(f"Training samples: {training_number}")
        print(f"Validating samples: {len(df)-training_number}")
        df_trn = df_shuffled[:training_number]
        df_dev = df_shuffled[training_number:]
        print(f"Training samples: {len(df_trn)}")
        print(f"Validating samples: {len(df_dev)}")
        df_trn.to_csv(config["general"].get("training_set"), index=False)
        df_dev.to_csv(config["general"].get("dev_set"), index=False)


def dataset_to_batch(config):
    bs = config["general"].getint("batch_size")
    df_trn = pd.read_csv(config["general"].get("training_set"))
    df_dev = pd.read_csv(config["general"].get("dev_set"))
    print(f"Spliiting for batchsize = {bs}")
    print(f"Training set has {len(df_trn)} samples, {len(df_trn) // bs + 1} batches")
    os.makedirs("dataset_split/arrays/training/batch/512/gray", exist_ok=True)
    os.makedirs("dataset_split/arrays/training/batch/512/rgb", exist_ok=True)
    for bn in range(len(df_trn) // bs + 1):
        labels = df_trn.iloc[bn * bs:(bn + 1) * bs]["labels"]
        file_names = df_trn.iloc[bn * bs:(bn + 1) * bs]["file_names"]
        batch = np.zeros((len(labels), 32, 32, 4), dtype=np.float32)
        batch[:, 0, 0, 3] = labels
        images = np.array(
            [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
        batch[:, :, :, 0:3] = images / 255.0
        np.save(f"dataset_split/arrays/training/batch/512/rgb/{bn}", batch)

        batch = np.zeros((len(labels), 32, 32, 2), dtype=np.float32)
        batch[:, 0, 0, 1] = labels

        images = np.array(
            [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x)).convert("L")) for x in
             file_names])
        batch[:, :, :, 0] = images / 255.0
        print(f"{np.max(batch)}/{np.min(batch)}/{np.mean(batch):.3f}/{np.std(batch):.3f}")
        np.save(f"dataset_split/arrays/training/batch/512/gray/{bn}", batch)
    print(f"Dev set has {len(df_dev)} samples, {len(df_dev) // bs + 1} batches")

    os.makedirs("dataset_split/arrays/dev/batch/512/gray", exist_ok=True)
    os.makedirs("dataset_split/arrays/dev/batch/512/rgb", exist_ok=True)
    for bn in range(len(df_dev) // bs + 1):
        labels = df_dev.iloc[bn * bs:(bn + 1) * bs]["labels"]
        file_names = df_dev.iloc[bn * bs:(bn + 1) * bs]["file_names"]
        batch = np.zeros((len(labels), 32, 32, 4), dtype=np.float32)
        batch[:, 0, 0, 3] = labels
        images = np.array(
            [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
        batch[:, :, :, 0:3] = images / 255.0
        np.save(f"dataset_split/arrays/dev/batch/512/rgb/{bn}", batch)

        batch = np.zeros((len(labels), 32, 32, 2), dtype=np.float32)
        batch[:, 0, 0, 1] = labels
        images = np.array(
            [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x)).convert("L")) for x in
             file_names])
        batch[:, :, :, 0] = images / 255.0
        np.save(f"dataset_split/arrays/dev/batch/512/gray/{bn}", batch)


def dataset_to_npy(config):
    df_trn = pd.read_csv(config["general"].get("training_set"))
    df_dev = pd.read_csv(config["general"].get("dev_set"))
    print(f"Spliiting for batchsize = {bs}")
    print(f"Training set has {len(df_trn)} samples")
    os.makedirs("dataset_split/arrays/training/", exist_ok=True)
    os.makedirs("dataset_split/arrays/training/", exist_ok=True)

    labels = df_trn["labels"]
    file_names = df_trn["file_names"]
    batch = np.zeros((len(labels), 32, 32, 4), dtype=np.float32)
    batch[:, 0, 0, 3] = labels
    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
    batch[:, :, :, 0:3] = images / 255.0
    np.save(f"dataset_split/arrays/training/rgb_all", batch)

    batch = np.zeros((len(labels), 32, 32, 2), dtype=np.float32)
    batch[:, 0, 0, 1] = labels

    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x)).convert("L")) for x in
         file_names])
    batch[:, :, :, 0] = images / 255.0
    print(f"{np.max(batch)}/{np.min(batch)}/{np.mean(batch):.3f}/{np.std(batch):.3f}")
    np.save(f"dataset_split/arrays/training/gray_all", batch)
    print(f"Dev set has {len(df_dev)} samples")

    os.makedirs("dataset_split/arrays/dev/", exist_ok=True)
    os.makedirs("dataset_split/arrays/dev/", exist_ok=True)

    labels = df_dev["labels"]
    file_names = df_dev["file_names"]
    batch = np.zeros((len(labels), 32, 32, 4), dtype=np.float32)
    batch[:, 0, 0, 3] = labels
    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
    batch[:, :, :, 0:3] = images / 255.0
    np.save(f"dataset_split/arrays/dev/rgb_all", batch)

    batch = np.zeros((len(labels), 32, 32, 2), dtype=np.float32)
    batch[:, 0, 0, 1] = labels
    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x)).convert("L")) for x in
         file_names])
    batch[:, :, :, 0] = images / 255.0
    np.save(f"dataset_split/arrays/dev/gray_all", batch)


if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    dataset_image(config)
    dataset_split(config)
    # dataset_to_batch(config)
    dataset_to_npy(config)
