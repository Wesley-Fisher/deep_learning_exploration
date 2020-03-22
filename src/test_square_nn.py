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


#
# Create Data
#
X = np.linspace(0, 2.0*np.pi, 1000)
Y = np.sin(X)
(X_train, X_test, Y_train, Y_test) = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

X_predictable = np.linspace(0, 2.0*np.pi, 50)
Y_truth = np.sin(X_predictable)



#
# Create Model
#
dists = DistrbutionTypes()
model = ScalarSquareNN(dists)
pd = model.get_parameter_descriptions()
mph_rand = ModelPerformanceHistory('square_nn_rand',
                                   list(pd.keys()),
                                   ['loss', 'val_loss'])
mph_gpr = ModelPerformanceHistory('square_nn_gpr',
                                  list(pd.keys()),
                                  ['loss', 'val_loss'])
ps_rand = ParamSampler(pd, mph_rand)
ps_gpr = ParamSampler(pd, mph_gpr)



#
# Training iterations - Random
#

# General Training
def hyper_training(model, ps, mph, ps_chooser):
    best_val_loss = None
    best_losses = None
    best_val_losses = None
    predicted = None
    for i in range(0, Num_Iterations):
        params = ps_chooser(ps)
        print(params)
        model.prepare_model(params)

        hist = model.train_model(X_train, Y_train,
                                X_test, Y_test,
                                100,
                                verbose=1)
        
        results = {    'loss': hist.history['loss'][-1],
                   'val_loss': hist.history['val_loss'][-1]}
        
        mph.add_sample(params, results)
        mph.save_history()

        loss = hist.history['val_loss'][-1]
        if i == 0 or loss < best_val_loss:
            best_val_loss = hist.history['val_loss'][-1]
            best_losses = hist.history['loss']
            best_val_losses = hist.history['val_loss']
            predicted = model.predict(X_predictable)
    
    return best_losses, best_val_losses, predicted


# Actual Training
rand_losses, rand_val_losses, yout_rand = hyper_training(model,
                                                         ps_rand,
                                                         mph_rand,
                                                         lambda ps: ps.sample_parameters_random())

gpr_losses, gpr_val_losses, yout_gpr = hyper_training(model,
                                                      ps_gpr,
                                                      mph_gpr,
                                                      lambda ps: ps.sample_parameters_gp_explore('val_loss'))



#
# Visualize and show results
#
fig = plt.figure(1)
plt.plot(rand_losses, label="Rand Train")
plt.plot(rand_val_losses, label="Rand Test")
plt.plot(gpr_losses, label="GPR Train")
plt.plot(gpr_val_losses, label="GPR Test")
plt.title("Square NN - Val. Losses over Training - Best Model")
plt.xlabel("Epochs")
plt.ylabel("Val. Loss  (Mean Squared)")
plt.legend()
plt.savefig("/workspace/results/plots/square_nn_01_best_losses")


fig = plt.figure(2)
plt.plot(X_predictable, Y_truth, label="Truth")
plt.plot(X_predictable, yout_rand, label="Random")
plt.plot(X_predictable, yout_gpr, label="GPR")
plt.title("Square NN - Best Network Predictions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.savefig("/workspace/results/plots/square_nn_02_best_predictions")


fig = plt.figure(3)
plt.plot(np.log10(mph_gpr.get_history('val_loss')), label='GPR')
plt.plot(np.log10(mph_rand.get_history('val_loss')), label='Rand')
plt.xlabel('Iteration (New Model and Params)')
plt.ylabel('Log10(Loss)  (Mean Squared)')
plt.title('Square NN - Val. Losses for Models Trained')
plt.legend()
plt.savefig("/workspace/results/plots/square_nn_03_all_losses")


fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
x = mph_gpr.get_history('num_hidden_dense')
print(x)
y = mph_gpr.get_history('num_neurons')
z = np.log10( mph_gpr.get_history('val_loss'))
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Log10(val_loss) in Parameter-Space - GPR Trained')
ax.set_xlabel('Num. Hidden Dense Layers')
ax.set_ylabel('Num. Neurons Per Layer')
ax.set_zlabel('Log10(Final Val. Loss)  (Mean Squared)')
plt.savefig("/workspace/results/plots/square_nn_04_losses_param_space")


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

#plt.show()