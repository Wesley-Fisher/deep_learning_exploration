#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sklearn
import sklearn.model_selection

from models.scalar_square_nn import ScalarSquareNN
from parameters.performance_history import ModelPerformanceHistory
from parameters.param_sampler import ParamSampler

# Create Data
X = np.linspace(0, np.pi, 100)
Y = np.sin(X)
(X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# Create Model
model = ScalarSquareNN()
pd = model.get_parameter_descriptions()
mph = ModelPerformanceHistory('square_nn',
                              list(pd.keys()),
                              ['loss', 'val_loss'])
ps = ParamSampler()

# Training iterations
for i in range(0, 10):

    # Choose parameters
    params = ps.sample_parameters(pd, 'random')

    model.prepare_model(params)

    # Train
    hist = model.train_model(X_train, Y_train,
                            X_test, Y_test,
                            500,
                            verbose=1)

    results = {    'loss': hist.history['loss'][-1],
               'val_loss': hist.history['val_loss'][-1]}
    
    mph.add_sample(params, results)
    mph.save_history()

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


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
x = mph.get_history('num_hidden_dense')
y = mph.get_history('num_neurons')
z =np.log10( mph.get_history('val_loss'))
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()