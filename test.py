import tensorflow as tf
import numpy as np
from utils.elpv_reader import load_dataset
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


images, proba, types = load_dataset()
images = images/255.
images = images.reshape(2624, 300, 300, 1)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), input_shape=(300, 300, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()


model = models.Sequential()

model.add(layers.Conv2D(64, kernel_size=3,
                        activation='relu', input_shape=(300, 300, 1)))
model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(images, proba)
