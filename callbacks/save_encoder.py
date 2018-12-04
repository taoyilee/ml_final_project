from keras.callbacks import Callback
from keras.models import Model
import os


class SaveEncoder(Callback):
    encoder: Model

    def __init__(self, encoder, output_dir):
        self.encoder = encoder
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        self.encoder.save(os.path.join(self.output_dir, f"encoder_{epoch+1:02d}.h5"))
