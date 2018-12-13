from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l1
import numpy as np


def autoencoder_model(color_mode="rgb", bottleneck_width=32, expand_ratio=2, hidden_layers=5, reg=None):
    print(f"Regularization coefficient is {reg}")
    input_shape = 32 * 32
    input_img = Input(shape=(input_shape,), name="input")
    layers = np.logspace(np.log2(input_shape * expand_ratio), np.log2(bottleneck_width), hidden_layers + 1, base=2)
    layers = [int(l) for l in layers[:-1]]
    print(f"Layer neurons = {layers}")
    x = input_img
    for i, l_width in enumerate(layers):
        kwargs = {'activation': 'relu', 'name': f"encoder{i}"}
        if i == 0 and reg is not None:
            kwargs.update({'activity_regularizer': l1(reg)})
        x = Dense(l_width, **kwargs)(x)

    kwargs = {}
    if reg is not None:
        kwargs = {'activity_regularizer': l1(reg)}
    bottleneck = Dense(bottleneck_width, activation='relu', name="bottleneck", **kwargs)(x)
    x = bottleneck
    for i, l_width in enumerate(layers[::-1]):
        kwargs = {'activation': 'relu', 'name': f"decoder{i}"}
        x = Dense(l_width, **kwargs)(x)
    x = Dense(input_shape, activation='sigmoid', name="output")(x)

    encoder = Model(input_img, bottleneck)
    return Model(input_img, x), encoder
