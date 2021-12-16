from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras import Model


class Autoencoder(Model, ABC):

    def __init__(self,
                 input_dim: tuple[int, int] = (28, 28),
                 outer_layer: int = 80,
                 middle_layer: int = 60,
                 inner_layer: int = 40,
                 latent_dim: int = 10):

        super(Autoencoder, self).__init__()

        self.input_dimensions = input_dim
        self.outer_layer = outer_layer
        self.inner_layer = inner_layer
        self.latent_dim = latent_dim

        # Create Encoder
        inputs = layers.Input(
            shape=(input_dim[0] * input_dim[1],),
            name='encoder_input'
        )

        enc_outer_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(outer_layer, activation='relu',
                                 name=f"enc_outer_{i}")(inputs)
            enc_outer_layers.append(layer)

        enc_middle_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(middle_layer, activation='relu',
                                 name=f"enc_middle_{i}")(
                enc_outer_layers[i]
            )
            enc_middle_layers.append(layer)

        enc_inner_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(inner_layer, activation='relu',
                                 name=f"enc_inner_{i}")(
                enc_middle_layers[i]
            )
            enc_inner_layers.append(layer)

        channels = list()
        for i in range(latent_dim):
            channel = layers.Dense(1, activation='softsign',
                                   name=f"channel_{i}")(
                enc_inner_layers[i]
            )
            channels.append(channel)

        self.encoder = Model(
            inputs = inputs,
            outputs = channels
        )

        # Create Decoder
        inputs = list()
        for i in range(latent_dim):
            layer = layers.Input(shape=(1,), name=f'decoder_input_{i}')
            inputs.append(layer)

        dec_inner_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(inner_layer, activation='relu',
                                 name=f"dec_inner_{i}")(inputs[i])
            dec_inner_layers.append(layer)

        dec_middle_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(outer_layer, activation='relu',
                                 name=f"dec_middle_{i}")(
                dec_inner_layers[i]
            )
            dec_middle_layers.append(layer)

        dec_outer_layers = list()
        for i in range(latent_dim):
            layer = layers.Dense(outer_layer, activation='relu',
                                 name=f"dec_outer_{i}")(
                dec_middle_layers[i]
            )
            dec_outer_layers.append(layer)

        joined_outer_layers = tf.concat(dec_outer_layers, axis=1)

        outputs = layers.Dense(
            input_dim[0]*input_dim[1],
            activation='sigmoid',
            name="dec_output"
        )(joined_outer_layers)

        self.decoder = Model(
            inputs = inputs,
            outputs = outputs
        )

    def call(self, x: np.array, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

