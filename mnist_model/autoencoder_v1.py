from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Autoencoder(Model, ABC):

    def __init__(self,
                 input_dimensions: tuple[int, int] = (28, 28),
                 outer_layer: int = 28 * 28,
                 inner_layer: int = 28 * 28,
                 latent_dim: int = 10):

        super(Autoencoder, self).__init__()

        self.input_dimensions = input_dimensions
        self.outer_layer = outer_layer
        self.inner_layer = inner_layer
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(
                input_dimensions[0] * input_dimensions[1],
                activation='relu'
            ),
            layers.Dense(outer_layer, activation='relu'),
            layers.Dense(inner_layer, activation='relu'),
            layers.Dense(latent_dim, activation='softsign')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(inner_layer, activation='relu'),
            layers.Dense(outer_layer, activation='relu'),
            layers.Dense(
                input_dimensions[0] * input_dimensions[1],
                activation='sigmoid'
            ),
            layers.Reshape((28, 28))
        ])

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
