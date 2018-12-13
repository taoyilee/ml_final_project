from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.layers import Reshape, UpSampling2D
from keras.models import Model, Sequential
from keras.regularizers import l1


def encoder(color_mode="grayscale", filter_size=3, filter_number=64, b_filter_number=2, convs=4,
            reg=None, batch_norm=True, droprate=0.2, pooling_size=(2, 2), weights=None):
    model = Sequential()

    model.add(
        Conv2D(filter_number, input_shape=(32, 32, 1 if color_mode == "grayscale" else 3), kernel_size=5,
               activation='relu', padding='same', activity_regularizer=l1(reg if reg is not None else 0), name="conv0"))
    model.add(BatchNormalization(name="bn0")) if batch_norm else None
    model.add(MaxPooling2D(pooling_size, name=f"pooling0"))
    for i in range(1, convs):
        model.add(Conv2D(filter_number, kernel_size=filter_size, activation='relu', padding='same',
                         activity_regularizer=l1(reg if reg is not None else 0), name=f"conv{i}"))
        model.add(BatchNormalization(name=f"bn{i}")) if batch_norm else None
        model.add(MaxPooling2D(pooling_size, name=f"pooling{i}"))
        model.add(Dropout(droprate))

    model.add(Conv2D(b_filter_number, filter_size, activation='sigmoid', padding="same", name="bottleneck_conv"))
    model.add(BatchNormalization(name="bottleneck_bn")) if batch_norm else None
    model.add(GlobalAveragePooling2D(name="bottleneck"))
    if weights is not None:
        model.load_weights(weights, skip_mismatch=True, by_name=True)
    return model


def decoder(input_shape=2, color_mode="rgb", filter_size=3, filter_number=32, convs=6,
            reg=None, batch_norm=True, droprate=0.2, weights=None):
    model = Sequential()
    model.add(Reshape((1, 1, input_shape), input_shape=(input_shape,), name="reshape"))
    for i in range(1, convs):
        model.add(UpSampling2D(2))
        model.add(Conv2D(filter_number, kernel_size=filter_size, activation='relu',
                         activity_regularizer=l1(reg if reg is not None else 0), padding='same', name=f"conv{i}"))
        model.add(BatchNormalization(name=f"bn{i}")) if batch_norm else None
        model.add(Dropout(droprate))

    model.add(Conv2D(3 if color_mode == "rgb" else 1, kernel_size=filter_size, activation='sigmoid',
                     activity_regularizer=l1(reg if reg is not None else 0), padding='same', name=f"conv_out"))
    model.add(BatchNormalization(name=f"bn_out")) if batch_norm else None
    model.add(Dropout(droprate))

    if weights is not None:
        model.load_weights(weights, skip_mismatch=True, by_name=True)
    return model


def svhn_ae(color_mode="grayscale", filter_size=3, filter_number=64, b_filter_number=4, convs=4,
            reg=None, batch_norm=True, droprate=0.2):
    ae_encoder = encoder(color_mode=color_mode, filter_size=filter_size, filter_number=filter_number,
                         b_filter_number=b_filter_number,
                         convs=convs, reg=reg, batch_norm=batch_norm, droprate=droprate)

    ae_decoder = decoder(input_shape=ae_encoder.output_shape[1], color_mode=color_mode, filter_size=filter_size,
                         filter_number=filter_number, reg=reg, convs=6, batch_norm=batch_norm, droprate=droprate)
    autoencoder = Model(inputs=ae_encoder.input, outputs=ae_decoder(ae_encoder.output))
    return autoencoder, ae_encoder
