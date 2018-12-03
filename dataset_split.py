from preprocessing.dataset import SVNHDataset
import os
import configparser as cp
import pandas as pd


def dataset_split(config):
    config.read("config.ini")
    training_ratio = config["general"].getfloat("train_ratio")
    if not os.path.exists("dataset_split/all.csv"):
        os.makedirs("dataset_split", exist_ok=True)
        train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
        file_names = train_set.save_for_viewing("dataset_split/images/training")
        df = pd.DataFrame()
        df["labels"] = train_set.labels.flatten()
        df["file_names"] = file_names
        df.to_csv("dataset_split/all.csv")
    df = pd.read_csv("dataset_split/all.csv", index_col=0)
    df_shuffled = df.sample(frac=1)
    print(df_shuffled[:10])

    training_number = int(len(df) * training_ratio / 100)
    print(f"Training samples: {training_number}")
    print(f"Validating samples: {len(df)-training_number}")
    df_trn = df_shuffled[:training_number]
    df_dev = df_shuffled[training_number:]
    print(f"Training samples: {len(df_trn)}")
    print(f"Validating samples: {len(df_dev)}")
    df_trn.to_csv(config["general"].get("training_set"))
    df_dev.to_csv(config["general"].get("dev_set"))


if __name__ == "__main__":
    config = cp.ConfigParser()
    dataset_split(config)
