#!/usr/bin/env python3
import numpy as np
import heapq
import copy
import statistics

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import sklearn
import sklearn.model_selection

import scipy.stats

from models.scalar_highd_nn import ScalarHighDNN
from parameters.performance_history import ModelPerformanceHistory
from parameters.param_sampler import ParamSampler
from parameters.param_distributions import DistrbutionTypes
'''
Further testing in a higher-dimensional parameter space
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
model = ScalarHighDNN(dists)
pd = model.get_parameter_descriptions()
mph_rand = ModelPerformanceHistory('highd_nn_rand',
                                   list(pd.keys()),
                                   ['loss', 'val_loss'])
mph_gpr = ModelPerformanceHistory('highd_nn_gpr',
                                  list(pd.keys()),
                                  ['loss', 'val_loss'])
ps_rand = ParamSampler(pd, mph_rand)
ps_gpr = ParamSampler(pd, mph_gpr)



#
# Training iterations - Random
#

# General Training
def hyper_training(model, ps, mph, ps_chooser):

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
        
        uuid = mph.add_sample(params, results)
        mph.save_history()
        model.save('highd_nn_' + str(uuid), hist.history)
    
    return


# Actual Training
hyper_training(model, ps_rand, mph_rand, lambda ps: ps.sample_parameters_random())

hyper_training(model, ps_gpr, mph_gpr, lambda ps: ps.sample_parameters_gp_explore('val_loss'))


#
# Collect Relevant Information
#
best_rand_uuid = mph_rand.get_best('val_loss', dir=-1)['uuid']
best_gpr_uuid = mph_gpr.get_best('val_loss', dir=-1)['uuid']

best_rand_train_hist = model.load('highd_nn_' + best_rand_uuid)
yout_rand = model.predict(X_predictable)


best_gpr_train_hist = model.load('highd_nn_' + best_gpr_uuid)
yout_gpr = model.predict(X_predictable)

gpr_loss_history = mph_gpr.get_history_of('val_loss')
rand_loss_history = mph_rand.get_history_of('val_loss')



#
# Visualize and show results
#

fig = plt.figure(1)
plt.plot(best_rand_train_hist['loss'], label="Rand Train")
plt.plot(best_rand_train_hist['val_loss'], label="Rand Test")
plt.plot(best_gpr_train_hist['loss'], label="GPR Train")
plt.plot(best_gpr_train_hist['val_loss'], label="GPR Test")
plt.title("HighD NN - Val. Losses over Training - Best Model")
plt.xlabel("Epochs")
plt.ylabel("Val. Loss  (Mean Squared)")
plt.legend()
plt.savefig("/workspace/results/major/highd_nn/highd_nn_01_best_losses")



fig = plt.figure(2)
plt.plot(X_predictable, Y_truth, label="Truth")
plt.plot(X_predictable, yout_rand, label="Random")
plt.plot(X_predictable, yout_gpr, label="GPR")
plt.title("HighD NN - Best Network Predictions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.savefig("/workspace/results/major/highd_nn/highd_nn_02_best_predictions")


fig = plt.figure(3)
plt.plot(np.log10(gpr_loss_history), label='GPR')
plt.plot(np.log10(rand_loss_history), label='Rand')
plt.xlabel('Iteration (New Model and Params)')
plt.ylabel('Log10(Loss)  (Mean Squared)')
plt.title('HighD NN - Val. Losses for Models Trained')
plt.legend()
plt.savefig("/workspace/results/major/highd_nn/highd_nn_03_all_losses")


def min_so_far(data):
    data = copy.copy(data)
    for i in range(1, len(data)):
        data[i] = min(data[i-1], data[i])
    return data

fig = plt.figure(4)
plt.plot(np.log10(min_so_far(gpr_loss_history)), label='GPR')
plt.plot(np.log10(min_so_far(rand_loss_history)), label='Rand')
plt.xlabel('Iteration (New Model and Params)')
plt.ylabel('Log10(Loss)  (Mean Squared)')
plt.title('HighD NN - Min-So-Far Val. Losses for Models Trained')
plt.legend()
plt.savefig("/workspace/results/major/highd_nn/highd_nn_04_min_so_far_losses")


with open("/workspace/results/major/highd_nn/highd_nn_05_statistics.txt", 'w') as f:
    f.write("HighD NN")
    rands = mph_rand.get_history_of('val_loss')
    gprs = mph_gpr.get_history_of('val_loss')

    t, p = scipy.stats.ttest_ind(rands, gprs, equal_var=False)
    f.write("Full (%d) Val-Loss Comparison:\n" % len(rands))
    f.write("  rand mean: %f\n" % statistics.mean(rands))
    f.write("  gpr mean: %f\n" % statistics.mean(gprs))
    f.write("  rand stdev: %f\n" % statistics.stdev(rands))
    f.write("  gpr stdev: %f\n" % statistics.stdev(gprs))
    f.write("  t value: %f\n" % t)
    f.write("  p value: %f\n" % p)

    rands = heapq.nsmallest(N_smallest, rands)
    gprs = heapq.nsmallest(N_smallest, gprs)

    t, p = scipy.stats.ttest_ind(rands, gprs, equal_var=False)
    f.write("Best %d Val-Loss Comparison:\n" % N_smallest)
    f.write("  rand mean: %f\n" % statistics.mean(rands))
    f.write("  gpr mean: %f\n" % statistics.mean(gprs))
    f.write("  rand stdev: %f\n" % statistics.stdev(rands))
    f.write("  gpr stdev: %f\n" % statistics.stdev(gprs))
    f.write("  t value: %f\n" % t)
    f.write("  p value: %f\n" % p)
