import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from mnist_model.autoencoder_v2 import Autoencoder
from mnist_model.tools import compare_input_output as compare

data_dir = 'mnist_model/data'
model_dir = './models'


def main():
    # build_and_store_models()
    get_mse_results()
    # examine_encoder_output()


def get_mse_results():
    """

    Runs the mnist_autoencoder for channels n in range [1, 20], 10 rounds each.
    Uses 10,000 values to train, 10,000 to validate, and 10,000 to test. Shuffles
    all sets of values between runs. Calculates MSE on test values - comparing
    input to output - and reports mean MSE.

    Exports data as dictionary to 'latent_dim_of_10.pickle'.

    """

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1]*x_train.shape[2]
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1]*x_test.shape[2]
    )

    data = {x: list() for x in range(1, 21)}

    for i in range(10):
        print(f"\n\n\nStarting Iteration {i+1} of 10...\n\n\n")
        for n in range(1, 21):

            np.random.shuffle(x_train)
            np.random.shuffle(x_test)
            x_data = x_train[:10_000]
            v_data = x_train[10_000:20_000]

            autoencoder = Autoencoder(
                latent_dim=n
            )

            autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

            dttm_fit_start = pd.Timestamp.now()
            print(f"Started Fitting Encoder, x={n}...")

            callback = EarlyStopping(monitor='val_loss', patience=10,
                                     restore_best_weights=True,
                                     min_delta=0.0001)

            autoencoder.fit(x_data, x_data,
                            epochs=1000, callbacks=[callback],
                            shuffle=False,
                            validation_data=(v_data, v_data))

            results = list()
            for i in x_test[:1000]:
                results.append(compare(autoencoder, i[np.newaxis, :]))
            data[n].append(np.mean(results))
            print(np.mean(results))

    with open('./data/v2_latent_dim_10_samples.pickle', 'wb') as file:
        pickle.dump(data, file)


def build_and_store_models():
    """

    Builds and trains models for n in [1, 21], then stores them in models
    directory. Uses full training data set.

    """

    (x_train, _), (_, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1]*x_train.shape[2]
    )

    for n in range(1, 21):

        np.random.shuffle(x_train)
        x_data = x_train[:50_000]
        v_data = x_train[50_000:]

        autoencoder = Autoencoder(
            latent_dim=n
        )

        autoencoder.compile(optimizer='adam',
                            loss=losses.MeanSquaredError())

        dttm_fit_start = pd.Timestamp.now()
        print(f"Started Fitting Encoder, n={n}, time={dttm_fit_start}...")

        callback = EarlyStopping(monitor='val_loss', patience=10,
                                 restore_best_weights=True,
                                 min_delta=0.0001)

        autoencoder.fit(x_data, x_data,
                        epochs=1000, callbacks=[callback],
                        shuffle=False,
                        validation_data=(v_data, v_data))

        save_file = os.path.join(model_dir, f"v2_n_{n}")
        autoencoder.save(save_file)


def examine_encoder_output():
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1]*x_test.shape[2]
    )

    grouped_images = {x: list() for x in range(10)}
    for i in range(len(x_test)):
        grouped_images[y_test[i]].append(x_test[i])

    file_path = os.path.join(
        model_dir, f"v1_n_10"
    )
    model = models.load_model(file_path)

    results = model.encoder.predict(np.expand_dims(x_test[0], 0))
    print(results)


if __name__ == '__main__':
    main()
