from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import os


def prepare_model(input_dim, nodes_per_hidden=2048, hidden_layers=3, regularization=0, lr=0.001, exp_dir="experiments",
                  weight_file=None):
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Sequential()
    model.add(Dense(units=nodes_per_hidden, activation='relu', kernel_regularizer=regularizers.l2(regularization),
                    input_dim=input_dim, name='input'))
    for h in range(hidden_layers):
        model.add(Dense(units=nodes_per_hidden, activation='relu',
                        kernel_regularizer=regularizers.l2(regularization), name=f"hidden_{h+1}"))
    model.add(Dense(units=2, activation='softmax',
                    kernel_regularizer=regularizers.l2(regularization), name='output_layer'))

    if weight_file is not None and os.path.isfile(weight_file):
        print(f"Loading previously trained weights from {weight_file}")
        model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    model_summary_file = f"{exp_dir}/model_summary.txt"
    with open(model_summary_file, "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    return model
