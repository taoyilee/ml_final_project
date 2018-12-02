from ae.ae_simple import autoencoder_model
from preprocessing.dataset import SVNHDataset
import numpy as np
from keras.callbacks import ReduceLROnPlateau

if __name__ == "__main__":
    batch_size = 512
    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    train_set.set_gray_scale()
    input_shape = np.prod(train_set.images.shape[1:])
    print(f"input_shape is {input_shape}")
    autoencoder, encoder = autoencoder_model(input_shape, 32)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    with open(f"weights/autoencoder_{train_set.color_mode}.json", "w+") as f:
        f.write(autoencoder.to_json())

    gen = train_set.generator(batch_size=batch_size, ae=True)
    callbacks = []
    callbacks.append(
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='min', min_delta=0.0001,
                          cooldown=0, min_lr=1e-10))
    autoencoder.fit_generator(gen, epochs=100, shuffle=True, callbacks=callbacks, verbose=1)
    autoencoder.save_weights(f"weights/autoencoder_{train_set.color_mode}.h5")
