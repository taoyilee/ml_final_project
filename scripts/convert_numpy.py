from preprocessing.dataset import SVHNDataset, SVHNPlotter, ColorConverter
import configparser as cp
from util.dataset import dataset_to_npy

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")

    train_set = SVHNDataset.from_mat("dataset/train_32x32.mat")
    print(train_set)
    plotter = SVHNPlotter(output_dir="images")
    plotter.save_mosaic(train_set, 10, 10)
    colorConv = ColorConverter(target_color="grayscale")
    train_set_gray = colorConv.transform(train_set)
    train_set_gray.name = train_set.name + "_gray"
    plotter.save_mosaic(train_set_gray, 10, 10)
    dataset_to_npy(config["general"].get("train_mat"))
    dataset_to_npy(config["general"].get("test_mat"))
    dataset_to_npy(config["general"].get("extra_mat"))
