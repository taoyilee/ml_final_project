import configparser as cp
from util.dataset import dataset_to_image_dir, dataset_split, dataset_to_batch, dataset_to_npy

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    dataset_to_image_dir(config)
    dataset_split(config)
    # dataset_to_batch(config)
    dataset_to_npy(config)
