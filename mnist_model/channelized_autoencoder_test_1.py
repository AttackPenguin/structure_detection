import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist

from mnist_model.channelized_autoencoder import Autoencoder
from mnist_model.tools import compare_input_output as compare


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

data = {x: list() for x in range(1, 21)}

for _ in range(1):
    for x in range(1, 21):

        np.random.shuffle(x_train)
        np.random.shuffle(x_test)
        x_data = x_train[:10_000]
        v_data = x_train[10_000:20_000]

        autoencoder = Autoencoder(
            latent_dim=x
        )

        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        autoencoder.build((28, 28))

        dttm_fit_start = pd.Timestamp.now()
        print(f"Started Fitting Encoder, x={x}...")

        callback = EarlyStopping(monitor='val_loss', patience=10,
                                 restore_best_weights=True,
                                 min_delta=0.0001)

        autoencoder.fit(x_data, x_data,
                        epochs=1000, callbacks=[callback],
                        shuffle=False,
                        validation_data=(v_data, v_data))

        # autoencoder.save_weights(f'./models/dim_10.h5')
        # autoencoder.build((1, 28, 28))
        # autoencoder.load_weights('./models/scratch_vsmall.h5')

        results = list()
        for i in x_test[:1000]:
            results.append(compare(autoencoder, i[np.newaxis, :, :]))
        data[x].append(np.mean(results))
        print(np.mean(results))

with open('./data/latent_dim_of_10.pickle', 'wb') as file:
    pickle.dump(data, file)
