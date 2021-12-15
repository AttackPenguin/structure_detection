import numpy as np

from mnist_model.autoencoder_v1 import Autoencoder


def generate_noise(
    dimensions: tuple[int, int] = (28, 28),
    count: int = 70_000
):

    data = np.random.rand(
        count, dimensions[0], dimensions[1]
    )
    return data


def compare_input_output(
    autoencoder: Autoencoder,
    input: np.ndarray
):

    input_f = input.flatten()
    output_f = autoencoder.predict(input).flatten()
    diff = ((input_f-output_f)**2).mean(axis=None)
    return diff
