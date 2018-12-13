import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


def sample_and_split(raw_data, raw_data_test, train_percentage=10, dev_percentage=10, rescale=True,
                     exp_dir="experiments"):
    split_indicator = f"{exp_dir}/.dataset_splitted"
    if os.path.isfile(split_indicator):
        print("Dataset already splitted, loading ...")
        x_trn = np.load(f"{exp_dir}/x_trn.npy")
        y_trn = np.load(f"{exp_dir}/y_trn.npy")
        x_dev = np.load(f"{exp_dir}/x_dev.npy")
        y_dev = np.load(f"{exp_dir}/y_dev.npy")
        x_test = np.load(f"{exp_dir}/x_test.npy")
        return x_trn, x_dev, y_trn, y_dev, x_test

    np.random.shuffle(raw_data)
    n_trn_data = np.int((train_percentage / 100) * raw_data.shape[0])
    n_dev_data = np.int((dev_percentage / 100) * raw_data.shape[0])
    x_trn = raw_data[:n_trn_data, :-1]
    x_dev = raw_data[n_trn_data:n_trn_data + n_dev_data, :-1]
    x_test = raw_data_test
    y_trn = raw_data[:n_trn_data, -1]
    y_dev = raw_data[n_trn_data:n_trn_data + n_dev_data, -1]
    print(f"Training with {x_trn.shape[0]}({100*x_trn.shape[0]/raw_data.shape[0]:.1f}%) points")
    print(f"Validating with {x_dev.shape[0]}({100*x_dev.shape[0]/raw_data.shape[0]:.1f}%) points")
    if rescale:
        print(f"Rescaling ENABLED")
        scaler = StandardScaler()
        scaler.fit(x_trn)
        with open(f"{exp_dir}/scaler.pickle", "wb") as fpickle:
            pickle.dump(scaler, fpickle)
        x_trn = scaler.transform(x_trn)
        x_dev = scaler.transform(x_dev)
        x_test = scaler.transform(x_test)
    else:
        print(f"Rescaling DISABLED")

    with open(split_indicator, 'a'):
        os.utime(split_indicator, None)

    print(f"shape of x_trn: {x_trn.shape}")
    print(f"shape of y_trn: {y_trn.shape}")
    print(f"shape of x_dev: {x_dev.shape}")
    print(f"shape of y_dev: {y_dev.shape}")
    print(f"shape of x_test: {x_test.shape}")
    np.save(f"{exp_dir}/x_trn", x_trn)
    np.save(f"{exp_dir}/y_trn", y_trn)
    np.save(f"{exp_dir}/x_dev", x_dev)
    np.save(f"{exp_dir}/y_dev", y_dev)
    np.save(f"{exp_dir}/x_test", x_test)
    return x_trn, x_dev, y_trn, y_dev, x_test
