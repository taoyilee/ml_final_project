from keras.layers import Input, Dense
from keras.models import Model


def autoencoder_model(input_shape=32 * 32, encoding_dim=32):
    # this is our input placeholder
    input_img = Input(shape=(input_shape,), name="input")
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu', name="encoder")(encoded)
    # "decoded" is the lossy reconstruction of the input
    encoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_shape, activation='sigmoid', name="decoder")(decoded)

    # this model maps an input to its reconstruction
    encoder = Model(input_img, encoded)
    return Model(input_img, decoded), encoder
