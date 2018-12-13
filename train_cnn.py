from util.cnn_trainer import prepare_callbacks, prepare_dataset, prepare_model
import configparser as cp
from datetime import datetime as dt
import os
from shutil import copyfile

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config_train_cnn.ini")
    batch_size = config["general"].getint("batch_size")
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    noise_ratio = config["general"].getfloat("noise_ratio")
    training_gen, validation_gen = prepare_dataset(config["general"].get("training_set"),
                                                   config["general"].get("dev_set"), color_mode=color_mode,
                                                   batch_size=batch_size, noise_ratio=noise_ratio)

    tag = dt.now().strftime("%m_%d_%H%M%S") + f"_{color_mode}_{ae_model}"
    log_dir, exp_dir = f"logs/{tag}", f"experiments/{tag}"
    os.makedirs(log_dir, exist_ok=True), os.makedirs(exp_dir, exist_ok=True)
    copyfile("config.ini", os.path.join(exp_dir, "config.ini"))
    print(f"experiment results will be written to {exp_dir}")

    callbacks = prepare_callbacks(early_stop=config["general"].getboolean("early_stop"),
                                  patience_lr=config["training"].getint("patientce_reduce_lr"),
                                  batch_size=batch_size, log_dir=log_dir,
                                  verbosity=config["training"].getint("verbosity"))
    cnn_model = prepare_model(lr=config["optimizer"].getfloat("lr"), exp_dir=exp_dir, color_mode=color_mode,
                              print_summary=True, droprate=config["svhn_cnn"].getfloat("droprate"),
                              batch_norm=config["svhn_cnn"].getboolean("batch_norm"),
                              fc_hidden=config["svhn_cnn"].getint("fc_hidden"),
                              filter_number=config["svhn_cnn"].getint("filter_number"))

    cnn_model.fit_generator(training_gen, validation_data=validation_gen, epochs=config["general"].getint("epoch"),
                            shuffle=True, callbacks=callbacks, verbose=config["training"].getint("verbosity"),
                            workers=4, use_multiprocessing=True)
    cnn_model.save_weights(os.path.join(exp_dir, f"cnn_final.h5"))
