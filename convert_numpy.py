from preprocessing.dataset import SVNHDataset

if __name__ == "__main__":
    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    print(train_set)
    train_set.save_for_viewing("output")
    train_set.to_greyscale()
    train_set.save_for_viewing("output")
