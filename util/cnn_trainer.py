from preprocessing.dataset import SVHNDataset, ColorConverter
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from keras import optimizers
from models.cnn.densenet import dense_net, svhn_cnn

import os


def prepare_callbacks(early_stop=True, exp_dir="experiments", log_dir="logs", patience_lr=2, batch_size=32,
                      verbosity=1):
    print("*** PREPARING CALLBACKS ***")
    callbacks = []
    callbacks.append(
        ModelCheckpoint(os.path.join(exp_dir, "full_weights.{epoch:02d}.h5"), monitor='val_loss', verbose=0,
                        save_best_only=False, save_weights_only=False, mode='auto', period=1))
    callbacks.append(CSVLogger(os.path.join(exp_dir, f"training.csv"), separator=',', append=False))
    callbacks.append(
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=verbosity, mode='min',
                          min_delta=1e-6, cooldown=0,
                          min_lr=1e-12))
    if early_stop:
        print("Early Stop enabled")
        callbacks.append(
            EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=5, verbose=1, mode='min',
                          baseline=None, restore_best_weights=False))
    else:
        print("Early Stop disabled")
    callbacks.append(
        TensorBoard(log_dir=log_dir, histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=True,
                    write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                    embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
    return callbacks


def prepare_dataset(trn_npy, dev_npy, noise_ratio=0.07, ae=False, color_mode="rgb", batch_size=32):
    print(f"*** PREPARING DATASET {color_mode}***")
    train_set = SVHNDataset.from_npy(trn_npy)
    dev_set = SVHNDataset.from_npy(dev_npy)
    if color_mode == "grayscale":
        print(f"Converting to grayscale")
        converter = ColorConverter()
        train_set = converter.transform(train_set)
        dev_set = converter.transform(dev_set)
    print(f"Training Set Color_Mode is {train_set.color_mode}")
    print(f"Dev Set Color_Mode is {train_set.color_mode}")

    input_shape = train_set.images_shape
    print(f"input_shape is {input_shape}")
    training_gen = train_set.generator(batch_size=batch_size, ae=ae, flatten=False, noise=noise_ratio)
    validation_gen = dev_set.generator(batch_size=batch_size, ae=ae, flatten=False, noise=noise_ratio)
    return training_gen, validation_gen


def prepare_model(lr=1e-4, exp_dir="experiments", print_summary=True, color_mode="grayscale", droprate=0.2,
                  batch_norm=True, fc_hidden=1024, filter_number=128):
    print(f"*** PREPARING CNN MODEL *** ")
    cnn_model = svhn_cnn(color_mode=color_mode, droprate=droprate, batch_norm=batch_norm, fc_hidden=fc_hidden,
                         filter_number=filter_number)
    adam = optimizers.adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                           amsgrad=False)

    cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    with open(os.path.join(exp_dir, f"cnn.json"), "w+") as f:
        f.write(cnn_model.to_json())

    if print_summary:
        cnn_model.summary()
    return cnn_model
