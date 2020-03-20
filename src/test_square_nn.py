#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sklearn
import sklearn.model_selection

from models.scalar_square_nn import ScalarSquareNN


# Create Data
X = np.linspace(0, np.pi, 100)
Y = np.sin(X)
(X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)



# Create Model
model = ScalarSquareNN()

pd = model.get_parameter_descriptions()

params = {}
for param in pd.keys():
    params[param] = pd[param][2]

model.prepare_model(params)

# Train
hist = model.train_model(X_train, Y_train,
                         X_test, Y_test,
                         1000)

# Visualize
plt.figure(1)
plt.plot(hist.history['loss'], label="Train Loss")
plt.plot(hist.history['val_loss'], label="Test Loss")
plt.title("Training and Validation Accuracy")
plt.legend()

Ypred = model.predict(X)
plt.figure(2)
plt.plot(X, Y, label="Truth")
plt.plot(X, Ypred, label="Predicted")
plt.title("Predicted Datas")
plt.legend()
plt.show()
