from keras.applications import DenseNet121
from keras.models import Model, Sequential

from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Conv2D


def dense_net(weights=None):
    bottleneck = DenseNet121(input_shape=(32, 32, 3), include_top=False,
                             weights=None)  # type: Model
    x = bottleneck.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', name="mlp_hidden0")(x)
    output = Dense(10, activation='sigmoid', name="output")(x)
    model = Model(inputs=bottleneck.input, outputs=output)
    return model


def svhn_cnn(weights=None, color_mode="grayscale"):
    model = Sequential()
    model.add(
        Conv2D(128, kernel_size=5, activation='relu', input_shape=(32, 32, 1 if color_mode == "grayscale" else 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu', name="mlp_hidden0"))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax', name="output"))

    return model
