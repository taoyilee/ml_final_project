from keras.applications import DenseNet121
from keras.models import Model, Sequential

from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Conv2D, Dropout


def dense_net(weights=None):
    bottleneck = DenseNet121(input_shape=(32, 32, 3), include_top=False,
                             weights=None)  # type: Model
    x = bottleneck.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu', name="mlp_hidden0")(x)
    output = Dense(10, activation='sigmoid', name="output")(x)
    model = Model(inputs=bottleneck.input, outputs=output)
    return model


def svhn_cnn(weights=None, color_mode="grayscale", droprate=0.2, batch_norm=True, fc_hidden=1024, filter_number=128):
    model = Sequential()
    model.add(
        Conv2D(filter_number, kernel_size=5, activation='relu',
               input_shape=(32, 32, 1 if color_mode == "grayscale" else 3), name="conv0"))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Dropout(droprate))

    model.add(Conv2D(filter_number, kernel_size=3, activation='relu', name="conv1"))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Dropout(droprate))

    model.add(Conv2D(filter_number, kernel_size=3, activation='relu', name="conv2"))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Dropout(droprate))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(fc_hidden, activation='relu', name="mlp_hidden0"))
    model.add(BatchNormalization()) if batch_norm else None
    model.add(Dropout(droprate))

    model.add(Dense(10, activation='softmax', name="output"))

    if weights is not None:
        model.load_weights(weights, skip_mismatch=True, by_name=True)
    return model
