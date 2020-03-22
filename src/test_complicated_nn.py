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

from models.scalar_complicated_nn import ScalarComplicatedNN
from parameters.performance_history import ModelPerformanceHistory
from parameters.param_sampler import ParamSampler
from parameters.param_distributions import DistrbutionTypes
'''
Used to do initial testing and validation of the GPR exploration process
Use simple 2-parameter network to visualize
Compare vs random to judge effectiveness
Plot performance, and use t-tests
'''

Num_Iterations = 25
N_smallest = 5


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
model = ScalarComplicatedNN(dists)
pd = model.get_parameter_descriptions()
mph_rand = ModelPerformanceHistory('complicated_nn_rand',
                                   list(pd.keys()),
                                   ['loss', 'val_loss'])
mph_gpr = ModelPerformanceHistory('complicated_nn_gpr',
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
        for j in range(0, 5):
            print("")

        params = ps_chooser(ps)
        print(params)

        model.prepare_model(params)

        hist = model.train_model(X_train, Y_train,
                                X_test, Y_test,
                                params['epochs'],
                                verbose=2)
        
        results = {    'loss': hist.history['loss'][-1],
                   'val_loss': hist.history['val_loss'][-1]}
        
        mph.add_sample(params, results)
        mph.save_history()

        loss = hist.history['val_loss'][-1]
        if i == 0 or loss < best_val_loss:
            best_val_loss = loss
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
plt.figure(1)
plt.plot(rand_losses, label="Rand Train")
plt.plot(rand_val_losses, label="Rand Test")
plt.plot(gpr_losses, label="GPR Train")
plt.plot(gpr_val_losses, label="GPR Test")
plt.title("Training and Validation Accuracy - Best Model")
plt.legend()


plt.figure(2)
plt.plot(X_predictable, Y_truth, label="Truth")
plt.plot(X_predictable, yout_rand, label="Random")
plt.plot(X_predictable, yout_gpr, label="GPR")
plt.title("Predicted Data from Best Networks")
plt.legend()


fig = plt.figure(3)
plt.plot(np.log10(mph_gpr.get_history('val_loss')), label='GPR')
plt.plot(np.log10(mph_rand.get_history('val_loss')), label='Rand')
plt.xlabel('Iteration')
plt.ylabel('Log10(Loss)')
plt.title('Final Validation Losses over Hyper-Iterations')
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