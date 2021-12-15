import os

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.callbacks import EarlyStopping

model_dir = './'

(train_images, train_labels), (test_images, test_labels) = \
    datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

val_images = train_images[:10_000]
train_images = train_images[10_000:]
val_labels = train_labels[:10_000]
train_labels = train_labels[10_000:]

class_names = list(range(10))

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

callback = EarlyStopping(monitor='val_loss', patience=10,
                         restore_best_weights=True,
                         min_delta=0.0001)

model.fit(train_images, train_labels,
          epochs=1000, callbacks=[callback],
          shuffle=True,
          validation_data=(val_images, val_labels))

score = model.evaluate(test_images, test_labels)
print(f"test loss: {score[0]}")
print(f"test accuracy: {score[1]}")

save_file = os.path.join(model_dir, 'classifier_v1')
model.save(save_file)
