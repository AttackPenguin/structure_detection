import os
import pickle

import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models

import parameters as p

sns.set_theme(context='paper')

fig_dir = os.path.join(p.ROOT_DIR, 'mnist_model/figures/autoencoder_v2')
data_dir = os.path.join(p.ROOT_DIR, 'mnist_model/data')
model_dir = os.path.join(p.ROOT_DIR, 'mnist_model/models')


def main():
    # plot_latent_dim_of_10_means()
    # plot_latent_dim_of_10_boxplots()
    # visualize_transformed_images()
    input = [0.0]*10
    input[9] = 1.0
    generate_decoder_op(
        10, input,
        True
    )


def plot_latent_dim_of_10_means():
    data_file = os.path.join(data_dir, 'v2_latent_dim_10_samples.pickle')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    ax.plot(list(data.keys()), np.mean(np.array(list(data.values())), axis=1))

    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title("MNIST mean MSE, Autoencoder v2, 10 samples per # of channels")
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("Mean MSE")

    fig.show()

    save_file = os.path.join(fig_dir, '01a_autoencoder_v2_mean_mse.png')
    fig.savefig(save_file)


def plot_latent_dim_of_10_boxplots():
    data_file = os.path.join(data_dir, 'v2_latent_dim_10_samples.pickle')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    ax.boxplot(list(data.values()))

    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title("MNIST MSE, Autoencoder v2, 10 samples per # of "
                 "channels")
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("MSE")

    fig.show()

    save_file = os.path.join(fig_dir, '01b_autoencoder_v2_boxplots_mse.png')
    fig.savefig(save_file)


def visualize_transformed_images():
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1]*x_test.shape[2]
    )

    grouped_images = {x: list() for x in range(10)}
    for i in range(len(x_test)):
        grouped_images[y_test[i]].append(x_test[i])

    for n in range(1, 21):

        file_path = os.path.join(
            model_dir, f"v2_n_{n}"
        )
        model = models.load_model(file_path)

        fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
        outer = gridspec.GridSpec(1, 10, wspace=0.5)

        for value in range(10):
            inner = outer[value].subgridspec(10, 2)
            for row in range(10):
                left = fig.add_subplot(inner[row, 0])  # type: plt.Axes
                right = fig.add_subplot(inner[row, 1])  # type: plt.Axes
                left.imshow(grouped_images[value][row].reshape((28, 28)),
                            cmap='Greys')
                right.imshow(
                    model.predict(
                        np.expand_dims(grouped_images[value][row], 0)
                    )[0].reshape((28, 28)), cmap='Greys'
                )
                left.grid(False)
                right.grid(False)
                left.set_xticklabels([])
                left.set_yticklabels([])
                right.set_xticklabels([])
                right.set_yticklabels([])

        fig.suptitle(f"Inputs / Outputs for {n} Channel(s), Autoencoder v2")
        fig.show()
        save_file = os.path.join(
            fig_dir, 'input-output', f'{n}_channels.png'
        )
        fig.savefig(save_file)


def generate_decoder_op(n: int,
                        input: list[float],
                        show: bool = False):

    file_path = os.path.join(
        model_dir, f"v2_n_{n}"
    )
    model = models.load_model(file_path)

    inputs = list()
    for i in range(len(input)):
        inputs.append(tf.constant([[input[i]]]))

    image = model.decoder(inputs)

    if show:
        plt.imshow(image[0].numpy().reshape((28, 28)), cmap='Greys')
        plt.show()
    pass


if __name__ == '__main__':
    main()
