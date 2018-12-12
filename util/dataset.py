from preprocessing.dataset import SVHNDataset, SVHNPlotter
import os
import pandas as pd
import numpy as np
import PIL.Image


def dataset_to_image_dir(config):
    file_name = config["general"].get("dataset_all")
    if not os.path.exists(file_name):
        os.makedirs("dataset_split", exist_ok=True)
        train_set = SVHNDataset.from_mat(config["general"].get("train_mat"))
        plotter = SVHNPlotter(output_dir="dataset_split/images/training")
        file_names = plotter.save_images(train_set)
        df = pd.DataFrame()
        df["labels"] = train_set.labels.flatten()
        df["file_names"] = file_names
        df.to_csv(file_name, index=False)


def dataset_split(config):
    df = pd.read_csv("dataset_split/all.csv")
    if not os.path.exists(config["general"].get("training_set_csv")) or not os.path.exists(
            config["general"].get("dev_set_csv")):
        df_shuffled = df.sample(frac=1)
        print(df_shuffled[:10])
        training_ratio = config["general"].getfloat("train_ratio")
        training_number = int(len(df) * training_ratio / 100)
        print(f"Training samples: {training_number}")
        print(f"Validating samples: {len(df) - training_number}")
        df_trn = df_shuffled[:training_number]
        df_dev = df_shuffled[training_number:]
        print(f"Training samples: {len(df_trn)}")
        print(f"Validating samples: {len(df_dev)}")
        df_trn.to_csv(config["general"].get("training_set_Csv"), index=False)
        df_dev.to_csv(config["general"].get("dev_set_csv"), index=False)


def dataset_to_batch(config):
    bs = config["general"].getint("batch_size")
    df_trn = pd.read_csv(config["general"].get("training_set_csv"))
    df_dev = pd.read_csv(config["general"].get("dev_set_csv"))
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
    training_set = config["general"].get("training_set_csv")
    print(f"Training set is {training_set}")
    dev_set = config["general"].get("dev_set_csv")
    print(f"Dev set is {dev_set}")
    df_trn = pd.read_csv(training_set)
    df_dev = pd.read_csv(dev_set)
    print(f"Training set has {len(df_trn)} samples")
    os.makedirs("dataset_split/arrays/training/", exist_ok=True)

    labels = df_trn["labels"]
    file_names = df_trn["file_names"]
    batch = np.zeros((len(labels), 32, 32, 4), dtype=np.uint8)
    batch[:, 0, 0, 3] = labels
    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
    batch[:, :, :, 0:3] = images
    print(f"{np.max(batch)}/{np.min(batch)}/{np.mean(batch)}/{np.std(batch)}")
    np.save(f"dataset_split/arrays/training/rgb_all", batch)

    print(f"Dev set has {len(df_dev)} samples")

    os.makedirs("dataset_split/arrays/dev/", exist_ok=True)

    labels = df_dev["labels"]
    file_names = df_dev["file_names"]
    batch = np.zeros((len(labels), 32, 32, 4), dtype=np.uint8)
    batch[:, 0, 0, 3] = labels
    images = np.array(
        [np.array(PIL.Image.open(os.path.join("dataset_split/images/training", x))) for x in file_names])
    batch[:, :, :, 0:3] = images
    print(f"{np.max(batch)}/{np.min(batch)}/{np.mean(batch)}/{np.std(batch)}")
    np.save(f"dataset_split/arrays/dev/rgb_all", batch)
