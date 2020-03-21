#!/usr/bin/env python3
import numpy as np
import heapq

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sklearn
import sklearn.model_selection

import scipy.stats

from models.scalar_square_nn import ScalarSquareNN
from parameters.performance_history import ModelPerformanceHistory
from parameters.param_sampler import ParamSampler
from parameters.param_distributions import DistrbutionTypes
'''
Used to do initial testing and validation of the GPR exploration process
Use simple 2-parameter network to visualize
Compare vs random to judge effectiveness
Plot performance, and use t-tests
'''

Num_Iterations = 50
N_smallest = 10

# Create Data
X = np.linspace(0, np.pi, 100)
Y = np.sin(X)
(X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


# Create Model
dists = DistrbutionTypes()
model = ScalarSquareNN(dists)
pd = model.get_parameter_descriptions()
mph_rand = ModelPerformanceHistory('square_nn_rand',
                                   list(pd.keys()),
                                   ['loss', 'val_loss'])
mph_gpr = ModelPerformanceHistory('square_nn_gpr',
                                  list(pd.keys()),
                                  ['loss', 'val_loss'])
ps = ParamSampler()



# Training iterations - Random
for i in range(0, Num_Iterations):
    # Choose parameters
    params = ps.sample_parameters_random(pd)
    model.prepare_model(params)

    # Train
    hist_rand = model.train_model(X_train, Y_train,
                            X_test, Y_test,
                            100,
                            verbose=1)

    results = {    'loss': hist_rand.history['loss'][-1],
               'val_loss': hist_rand.history['val_loss'][-1]}
    
    mph_rand.add_sample(params, results)
    mph_rand.save_history()

# Training iterations - GPR
for i in range(0, Num_Iterations):
    # Choose parameters
    params = ps.sample_parameters_gp_explore(pd, mph_gpr, 'val_loss')
    model.prepare_model(params)

    # Train
    hist_gpr = model.train_model(X_train, Y_train,
                            X_test, Y_test,
                            100,
                            verbose=1)

    results = {    'loss': hist_gpr.history['loss'][-1],
               'val_loss': hist_gpr.history['val_loss'][-1]}
    
    mph_gpr.add_sample(params, results)
    mph_gpr.save_history()

# Visualize
plt.figure(1)
plt.plot(hist_gpr.history['loss'], label="Train Loss")
plt.plot(hist_gpr.history['val_loss'], label="Test Loss")
plt.title("Training and Validation Accuracy - GPR Latest")
plt.legend()

Ypred = model.predict(X)
plt.figure(2)
plt.plot(X, Y, label="Truth")
plt.plot(X, Ypred, label="Predicted")
plt.title("Predicted Data - GPR Latest")
plt.legend()


fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
x = mph_gpr.get_history('num_hidden_dense')
y = mph_gpr.get_history('num_neurons')
z = np.log10( mph_gpr.get_history('val_loss'))
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Log10(val_loss) in Parameter-Space - GPR Trained')


fig = plt.figure(4)
plt.plot(np.log10(mph_gpr.get_history('val_loss')), label='GPR')
plt.plot(np.log10(mph_rand.get_history('val_loss')), label='Rand')
plt.xlabel('Iteration')
plt.ylabel('Log10(Loss)')
plt.legend()


rands = mph_rand.get_history('val_loss')
gprs = mph_gpr.get_history('val_loss')

t, p = scipy.stats.ttest_ind(rands, gprs, equal_var=False)
print("Full Data Comparison:")
print("  t value: %f" % t)
print("  p value: %f" % p)

rands = heapq.nsmallest(N_smallest, rands)
gprs = heapq.nsmallest(N_smallest, gprs)

t, p = scipy.stats.ttest_ind(rands, gprs, equal_var=False)
print("Best Data Comparison:")
print("  t value: %f" % t)
print("  p value: %f" % p)

plt.show()