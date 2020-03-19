#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.metrics import categorical_accuracy

from tensorflow.python.client import device_lib

import sklearn
import sklearn.model_selection


print(keras.backend.tensorflow_backend._get_available_gpus())
print(device_lib.list_local_devices())

FACTOR = 1.15

# Create Data
X = np.linspace(0, 100, 250)
Y = X * FACTOR
(X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Create Model
input_layer = Input(shape=(1,))
l1 =Dense(2, activation='relu')(input_layer)
l2 =Dense(2, activation='relu')(l1)
output_layer = Dense(1)(l2)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = keras.optimizers.Adam(lr=0.01,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=None,
                                  decay=0.0,
                                  amsgrad=False)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mean_squared_error'])
model.summary()

# Train
hist = model.fit(x=X_train, y=Y_train,
                 validation_data=(X_test, Y_test),
                 verbose=1, callbacks=None,  shuffle=True,
                 batch_size=16, epochs=500)
print(hist.history.keys())

plt.figure(1)
plt.plot(hist.history["acc"], label="Train Acc")
plt.plot(hist.history['val_acc'], label="Test Acc")
#plt.plot(hist.history['loss'], label="Train Loss")
#plt.plot(hist.history['val_loss'], label="Test Loss")
plt.title("Training and Validation Accuracy")
plt.legend()

Ypred = model.predict(X)
plt.figure(2)
plt.plot(X, label="Truth")
plt.plot(Ypred, label="Predicted")
plt.title("Predicted Datas")
plt.legend()
plt.show()
