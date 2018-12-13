from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from models.common import sample_and_split
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
from models.auc_callback import auc_callback
from keras import regularizers
import random
import string
import os
import pandas as pd
import pickle

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 30
    regularization = 0
    dev_p = 4
    runs = 1
    hidden_layers = 4
    nodes_per_hidden = [256, 512, 1024, 2048, 4096]
    df = pd.DataFrame()

    for r in range(runs):
        for hl in range(1, hidden_layers + 1):
            for n_hl in nodes_per_hidden:
                tag = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(5)])
                print(f"Run tag is {tag}")
                work_dir = f"run_{tag}"
                exp_dir = f"./experiments/{work_dir}"
                os.makedirs(f"./logs/{work_dir}", exist_ok=True)
                os.makedirs(exp_dir, exist_ok=True)
                x_trn_raw, x_dev_raw, y_trn, y_dev, scaler = sample_and_split(raw_data, train_percentage=trn_p,
                                                                              dev_percentage=dev_p,
                                                                              rescale=True)
                np.save(f"{exp_dir}/x_trn_raw", x_trn_raw)
                np.save(f"{exp_dir}/x_dev_raw", x_dev_raw)
                np.save(f"{exp_dir}/y_trn", y_trn)
                np.save(f"{exp_dir}/y_dev", y_dev)
                with open(f"{exp_dir}/scaler.pickle", "wb") as fpickle:
                    pickle.dump(scaler, fpickle)

                y_trn_encoded = to_categorical(y_trn)
                y_dev_encoded = to_categorical(y_dev)
                print(f"y_trn_encoded shape: {y_trn_encoded.shape}")
                print(f"y_trn_encoded[:10]: {y_trn_encoded[:10]}")

                optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
                model = Sequential()
                print(f"Input dimension ={x_trn_raw.shape[1]}")
                model.add(
                    Dense(units=n_hl, activation='relu', kernel_regularizer=regularizers.l2(regularization),
                          input_dim=x_trn_raw.shape[1]))
                for h in range(hl):
                    model.add(Dense(units=n_hl, activation='relu', kernel_regularizer=regularizers.l2(regularization)))
                model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(regularization)))

                callbacks = []
                callbacks.append(auc_callback(x_trn_raw, y_trn_encoded, x_dev_raw, y_dev_encoded))
                callbacks.append(
                    TensorBoard(log_dir=f"./logs/{work_dir}", histogram_freq=0, batch_size=32, write_graph=True,
                                write_grads=False,
                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                embeddings_metadata=None, embeddings_data=None, update_freq='epoch'))
                callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='min',
                                                   min_delta=1e-6, cooldown=0, min_lr=0))

                callbacks.append(
                    EarlyStopping(monitor='last_dev_auc', min_delta=1e-7, patience=5, verbose=1, mode='max',
                                  baseline=None, restore_best_weights=False))
                callbacks.append(CSVLogger(f"{exp_dir}/epoch_log.csv"))
                model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                              metrics=['categorical_accuracy'])
                model.summary()
                model_summary_file = f"{exp_dir}/model_summary.txt"
                with open(model_summary_file, "w+") as f:
                    model.summary(print_fn=lambda x: f.write(x + '\n'))

                history = model.fit(x_trn_raw, y_trn_encoded, validation_data=(x_dev_raw, y_dev_encoded), epochs=100,
                                    verbose=2, batch_size=1024, callbacks=callbacks)
                history.history["tag"] = tag
                history.history["hidden_layers"] = hl
                history.history["nodes_per_hidden"] = n_hl
                history.history["train_percent"] = trn_p
                history.history["dev_percent"] = dev_p
                for k, v in history.history.items():
                    if isinstance(history.history[k], list):
                        history.history[k] = history.history[k][-1]
                df = df.append(history.history, ignore_index=True)

                model.save(f"learners/dnn/hw4_model_{tag}.h5")

                Xte = np.genfromtxt('data/X_test.txt', delimiter=None)
                Xte_xfm = scaler.transform(Xte)
                Yte = np.vstack((np.arange(Xte.shape[0]), model.predict(Xte)[:, 1])).T
                np.savetxt(f"{exp_dir}/Y_submit.txt", Yte, ' %d, %.4f ', header=' ID,Prob1 ', comments='',
                           delimiter=',')
                df.to_csv("problem_5_summary.csv")
