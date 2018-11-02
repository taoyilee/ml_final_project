import os
from multiprocessing import Process
import wget
import time
import sys
import tarfile


def download_report_status(tar_url, output_file, dataset_dir="dataset"):
    start_time = time.time()
    p = Process(target=download_tar, args=(tar_url, output_file, dataset_dir))
    p.start()
    time.sleep(1)
    while p.exitcode is None:
        d = os.listdir(dataset_dir)
        for tempfile in d:
            if tempfile[:len(output_file)] == output_file:
                size = os.stat(os.path.join(dataset_dir, tempfile)).st_size / 1e6
                time_elapsed = time.time() - start_time
                speed = size / time_elapsed
                print(f"{output_file}: {size:.1f}MB ({speed:.2f}MB/s, {time_elapsed:.1f}s)")
            time.sleep(0.5)


def download_tar(tar_url, output_file, dataset_dir="dataset"):
    output_file = os.path.join(dataset_dir, output_file)
    if not os.path.isfile(output_file):
        print(f"Downloading {output_file} from {tar_url}")
        wget.download(tar_url, output_file)
        sys.exit(0)
    else:
        print(f"File {output_file} exists, skipping")
        sys.exit(1)


def fork_download(tar_url, output_file, dataset_dir="dataset"):
    p = Process(target=download_report_status, args=(tar_url, output_file, dataset_dir))
    p.start()
    return p


def extract_dataset(tarball, dataset_dir="dataset"):
    print(f"Extracting {tarball}")
    tar = tarfile.open(os.path.join(dataset_dir, tarball))
    tar.extractall(path=dataset_dir)
    tar.close()


if __name__ == '__main__':
    dataset_dir = "dataset"
    d = os.listdir(dataset_dir)
    for file in d:
        if file[-4:] == ".tmp":
            print(f"Deleting {os.path.join(dataset_dir, file)}")
            os.remove(os.path.join(dataset_dir, file))
    os.makedirs(dataset_dir, exist_ok=True)
    train_gz = "train.tar.gz"
    test_gz = "test.tar.gz"
    extra_gz = "extra.tar.gz"
    train_url = f"http://ufldl.stanford.edu/housenumbers/{train_gz}"
    test_url = f"http://ufldl.stanford.edu/housenumbers/{test_gz}"
    extra_url = f"http://ufldl.stanford.edu/housenumbers/{extra_gz}"
    print("Please be patient, the datasets are downloading...")
    fork_download(train_url, train_gz, dataset_dir=dataset_dir)
    fork_download(test_url, test_gz, dataset_dir=dataset_dir)
    fork_download(extra_url, extra_gz, dataset_dir=dataset_dir)
    extract_dataset(train_gz, dataset_dir=dataset_dir)
    extract_dataset(test_gz, dataset_dir=dataset_dir)
    extract_dataset(extra_gz, dataset_dir=dataset_dir)
