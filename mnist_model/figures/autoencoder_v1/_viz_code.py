import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_theme(context='paper')

fig_dir = 'mnist_model/figures/autoencoder_v1'
data_dir = 'mnist_model/data'


def main():
    plot_latent_dim_of_10_means()
    plot_latent_dim_of_10_boxplots()


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


if __name__ == '__main__':
    main()
