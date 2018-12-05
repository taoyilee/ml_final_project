from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.regularizers import l1


def ae_cnn_layer(input_shape=(32, 32, 1), filter_size=(3, 3), filter_number=16, b_filter_number=4, convs=4,
                 reg=None):
    print(f"CNN regularization is {reg}")
    print(f"CNN input_shape is {input_shape}")
    input_img = Input(shape=input_shape, name="input")
    pooling_size = (2, 2)
    x = input_img
    for i in range(convs - 1):
        kwargs = {'padding': 'same'}
        if i == 0 and reg is not None:
            kwargs.update({'activity_regularizer': l1(reg)})
        x = Conv2D(filter_number, filter_size, name=f"encoder_{i}", activation='relu', **kwargs)(x)
        x = MaxPooling2D(pooling_size, name=f"pooling_{i}")(x)

    x = Conv2D(b_filter_number, filter_size, activation='relu', padding="same", name="bottle_neck_conv")(x)
    x = MaxPooling2D(pooling_size, name="bottle_neck")(x)
    bottleneck = x
    for i in range(convs):
        x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_{i}")(x)
        x = UpSampling2D(pooling_size, name=f"upsamp_{i}")(x)

    x = Conv2D(input_shape[2], filter_size, activation='sigmoid', padding='same', name="output")(x)
    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder


def ae_cnn_3_layer(input_shape=(32, 32, 1), filter_size=(3, 3), filter_number=16, reg=None):
    print(f"CNN regularization is {reg}")
    print(f"CNN input_shape is {input_shape}")
    input_img = Input(shape=input_shape, name="input")

    x = Conv2D(16, filter_size, activation='relu', padding='same', name=f"encoder_1")(input_img)
    x = MaxPooling2D((2, 2), name=f"pooling_1")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"encoder_2")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_2")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"encoder_3")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_3")(x)

    bottleneck = x
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"decoder_1")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_1")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"decoder_2")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_2")(x)
    x = Conv2D(16, filter_size, activation='relu', padding='same', name=f"decoder_3")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_3")(x)
    x = Conv2D(input_shape[2], filter_size, activation='sigmoid', padding='same', name=f"output")(x)

    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder


def ae_cnn_4_layer(input_shape=(32, 32, 1), filter_size=(3, 3), filter_number=16, reg=None):
    print(f"CNN regularization is {reg}")
    print(f"CNN input_shape is {input_shape}")
    input_img = Input(shape=input_shape, name="input")

    x = Conv2D(16, filter_size, activation='relu', padding='same', name=f"encoder_1", activity_regularizer=l1(1e-7))(
        input_img)
    x = MaxPooling2D((2, 2), name=f"pooling_1")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"encoder_2")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_2")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"encoder_3")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_3")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"encoder_4")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_4")(x)

    bottleneck = x
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"decoder_1")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_1")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"decoder_2")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_2")(x)
    x = Conv2D(8, filter_size, activation='relu', padding='same', name=f"decoder_3")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_3")(x)
    x = Conv2D(16, filter_size, activation='relu', padding='same', name=f"decoder_4")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_4")(x)
    x = Conv2D(input_shape[2], filter_size, activation='sigmoid', padding='same', name=f"output")(x)

    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder


def ae_cnn_5_layer(input_shape=(32, 32, 1), filter_size=(3, 3), filter_number=16, reg=None):
    print(f"CNN regularization is {reg}")
    print(f"CNN input_shape is {input_shape}")
    input_img = Input(shape=input_shape, name="input")

    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_1")(input_img)
    x = MaxPooling2D((2, 2), name=f"pooling_1")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_2")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_2")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_3")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_3")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_4")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_4")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_5")(x)
    x = MaxPooling2D((2, 2), name=f"pooling_5")(x)

    bottleneck = x
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_1")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_1")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_2")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_2")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_3")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_3")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_4")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_4")(x)
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_5")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_5")(x)
    x = Conv2D(input_shape[2], filter_size, activation='sigmoid', padding='same', name=f"output")(x)

    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder


def ae_cnn_1_layer(input_shape=(32, 32, 1), filter_size=(5, 5), filter_number=16, reg=None):
    print(f"CNN regularization is {reg}")
    print(f"CNN input_shape is {input_shape}")
    input_img = Input(shape=input_shape, name="input")

    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"encoder_1")(input_img)
    x = MaxPooling2D((2, 2), name=f"pooling_1")(x)
    bottleneck = x
    x = Conv2D(filter_number, filter_size, activation='relu', padding='same', name=f"decoder_1")(x)
    x = UpSampling2D((2, 2), name=f"upsamp_1")(x)
    x = Conv2D(input_shape[2], filter_size, activation='sigmoid', padding='same', name=f"output")(x)

    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder
