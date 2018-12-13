import os


def split_path(full_path):
    directory, base_path = os.path.split(full_path)
    file_name, ext = os.path.splitext(base_path)
    return file_name, directory, ext, base_path
