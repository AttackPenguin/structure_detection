from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras import Model


class Autoencoder(Model, ABC):

    def __init__(self,
                 input_dimensions: tuple[int, int] = (28, 28),
                 outer_layer: int = 40,
                 inner_layer: int = 20,
                 latent_dim: int = 10):

        super(Autoencoder, self).__init__()

        self.input_dimensions = input_dimensions
        self.outer_layer = outer_layer
        self.inner_layer = inner_layer
        self.latent_dim = latent_dim

        # Create Encoder
        inputs = layers.Input(input_dimensions)

        enc_outer_layers = list()
        for _ in range(latent_dim):
            layer = layers.Dense(outer_layer, activation='relu')(inputs)
            enc_outer_layers.append(layer)

        enc_inner_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(inner_layer, activation='relu')
            enc_inner_layers.append(layer(enc_outer_layers[i]))

        channels = list()
        for i in range(latent_dim):
            channel = layers.Dense(1, activation='softsign')
            channels.append(channel(enc_inner_layers[i]))

        dec_inner_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(inner_layer, activation='relu')
            dec_inner_layers.append(layer(channels[i]))

        dec_outer_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(outer_layer, activation='relu')
            dec_outer_layers.append(layer(dec_inner_layers[i]))

        outputs = \
            layers.Dense((input_dimensions[0]*input_dimensions[1]), activation='sigmoid')(
                tf.concat(dec_outer_layers, axis=1)
            )

        self.model = Model(inputs=inputs, outputs=outputs, name='mnist_wide_channels')

    def call(self, x, **kwargs):
        output = self.model(x)
        reshape = layers.Reshape(self.input_dimensions)
        return reshape(output)

