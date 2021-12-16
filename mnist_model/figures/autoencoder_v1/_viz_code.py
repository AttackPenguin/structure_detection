import os
import pickle

import numpy as np
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models

sns.set_theme(context='paper')

fig_dir = 'mnist_model/figures/autoencoder_v1'
data_dir = 'mnist_model/data'
model_dir = 'mnist_model/models'


def main():
    # plot_latent_dim_of_10_means()
    # plot_latent_dim_of_10_boxplots()
    # visualize_transformed_images()
    pca_visualization()


def plot_latent_dim_of_10_means():
    data_file = os.path.join(data_dir, 'latent_dim_of_10.pickle')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    ax.plot(list(data.keys()), np.mean(np.array(list(data.values())), axis=1))

    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title("MNIST mean MSE, Autoencoder v1, 10 samples per # of channels")
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("Mean MSE")

    fig.show()

    save_file = os.path.join(fig_dir, '01a_autoencoder_v1_mean_mse.png')
    fig.savefig(save_file)


def plot_latent_dim_of_10_boxplots():
    data_file = os.path.join(data_dir, 'latent_dim_of_10.pickle')
    with open(data_file, 'rb') as file:
        data = pickle.load(file)

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    ax.boxplot(list(data.values()))

    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title("MNIST MSE, Autoencoder v1, 10 samples per # of "
                 "channels")
    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("MSE")

    fig.show()

    save_file = os.path.join(fig_dir, '01b_autoencoder_v1_boxplots_mse.png')
    fig.savefig(save_file)


def visualize_transformed_images():
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype('float32') / 255.

    grouped_images = {x: list() for x in range(10)}
    for i in range(len(x_test)):
        grouped_images[y_test[i]].append(x_test[i])

    for n in range(1, 21):

        file_path = os.path.join(
            model_dir, f"v1_n_{n}"
        )
        model = models.load_model(file_path)

        fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
        outer = gridspec.GridSpec(1, 10, wspace=0.5)

        for value in range(10):
            inner = outer[value].subgridspec(10, 2)
            for row in range(10):
                left = fig.add_subplot(inner[row, 0])  # type: plt.Axes
                right = fig.add_subplot(inner[row, 1])  # type: plt.Axes
                left.imshow(grouped_images[value][row], cmap='Greys')
                right.imshow(
                    model.predict(
                        np.expand_dims(grouped_images[value][row], 0)
                    )[0], cmap='Greys'
                )
                left.grid(False)
                right.grid(False)
                left.set_xticklabels([])
                left.set_yticklabels([])
                right.set_xticklabels([])
                right.set_yticklabels([])

        fig.suptitle(f"Inputs / Outputs for {n} Channel(s), Autoencoder v1")
        fig.show()
        save_file = os.path.join(
            fig_dir, 'input-output', f'{n}_channels.png'
        )
        fig.savefig(save_file)


def pca_visualization():
    (x, y), (_, _) = mnist.load_data()

    x = x.astype('float32') / 255.
    x = x.reshape(x.shape[0], (x.shape[1] * x.shape[2]))

    pca = PCA(n_components=2)
    x_r = pca.fit(x).transform(x)

    colors = ['C' + str(x) for x in range(10)]
    target_names = [str(x) for x in range(10)]

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()

    for color, val, name in zip(colors, range(10), target_names):
        plt.scatter(
            x_r[y == val, 0], x_r[y == val, 1],
            color=color, alpha=0.3, s=1, label=name
        )

    ax.set_title("2D PCA Analysis of training data")
    leg = ax.legend(markerscale=5)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    fig.show()

    save_file = os.path.join(fig_dir, 'pca analysis.png')
    fig.savefig(save_file)


if __name__ == '__main__':
    main()
