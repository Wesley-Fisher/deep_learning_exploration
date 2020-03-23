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

from models.scalar_square_nn import ScalarSquareNN
from models.scalar_highd_nn import ScalarHighDNN
from parameters.performance_history import ModelPerformanceHistory
from parameters.param_sampler import ParamSampler
from parameters.param_distributions import DistrbutionTypes
from utils.directories import Directories as DIR
'''
Used to do initial testing and validation of the GPR exploration process
Compare vs random to judge effectiveness
Plot performance, and use t-tests

a) simple 2-parameter network to visualize
b) higher-d network
'''



def run_nn_tests(modeltype, Num_Iterations, N_smallest):
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
    model = modeltype(dists)
    pd = model.get_parameter_descriptions()
    mph_rand = ModelPerformanceHistory(model.get_prefix() + '_rand',
                                    list(pd.keys()),
                                    ['loss', 'val_loss'])
    mph_gpr = ModelPerformanceHistory(model.get_prefix() + '_gpr',
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
                                    verbose=2)
            
            results = {    'loss': hist.history['loss'][-1],
                    'val_loss': hist.history['val_loss'][-1]}
            
            uuid = mph.add_sample(params, results)
            mph.save_history()
            model.save(DIR.TRAINED + model.get_prefix() + '_' + str(uuid))
        
        return


    # Actual Training
    hyper_training(model, ps_rand, mph_rand, lambda ps: ps.sample_parameters_random())

    hyper_training(model, ps_gpr, mph_gpr, lambda ps: ps.sample_parameters_gp_explore('val_loss'))


    #
    # Collect Relevant Information
    #
    best_rand_uuid = mph_rand.get_best('val_loss', dir=-1)['uuid']
    best_gpr_uuid = mph_gpr.get_best('val_loss', dir=-1)['uuid']

    best_rand_train_hist = model.load(DIR.TRAINED + model.get_prefix() + '_' + best_rand_uuid)
    model.save(DIR.MAJOR + model.get_prefix() + '/' + model.get_prefix() + '_' + best_rand_uuid)
    yout_rand = model.predict(X_predictable)


    best_gpr_train_hist = model.load(DIR.TRAINED + model.get_prefix() + '_' + best_rand_uuid)
    model.save(DIR.MAJOR + model.get_prefix() + '/' + model.get_prefix() + '_' + best_rand_uuid)
    yout_gpr = model.predict(X_predictable)

    gpr_loss_history = mph_gpr.get_history_of('val_loss')
    rand_loss_history = mph_rand.get_history_of('val_loss')



    #
    # Visualize and show results
    #
    file_prefix = DIR.MAJOR + "/" + model.get_prefix() + "/" + model.get_prefix() + "_"

    fig = plt.figure(1)
    plt.plot(best_rand_train_hist['loss'], label="Rand Train")
    plt.plot(best_rand_train_hist['val_loss'], label="Rand Test")
    plt.plot(best_gpr_train_hist['loss'], label="GPR Train")
    plt.plot(best_gpr_train_hist['val_loss'], label="GPR Test")
    plt.title("Square NN - Val. Losses over Training - Best Model")
    plt.xlabel("Epochs")
    plt.ylabel("Val. Loss  (Mean Squared)")
    plt.legend()
    plt.savefig(file_prefix + "01_best_losses")



    fig = plt.figure(2)
    plt.plot(X_predictable, Y_truth, label="Truth")
    plt.plot(X_predictable, yout_rand, label="Random")
    plt.plot(X_predictable, yout_gpr, label="GPR")
    plt.title("Square NN - Best Network Predictions")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.savefig(file_prefix + "02_best_predictions")


    fig = plt.figure(3)
    plt.plot(np.log10(gpr_loss_history), label='GPR')
    plt.plot(np.log10(rand_loss_history), label='Rand')
    plt.xlabel('Iteration (New Model and Params)')
    plt.ylabel('Log10(Loss)  (Mean Squared)')
    plt.title('Square NN - Val. Losses for Models Trained')
    plt.legend()
    plt.savefig(file_prefix + "03_all_losses")


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
    plt.title('Square NN - Min-So-Far Val. Losses for Models Trained')
    plt.legend()
    plt.savefig(file_prefix + "04_min_so_far_losses")


    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    x = mph_gpr.get_history_of('num_hidden_dense')
    y = mph_gpr.get_history_of('num_neurons')
    z = np.log10( mph_gpr.get_history_of('val_loss'))
    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Square NN - Log10(val_loss) in Parameter-Space - GPR Trained')
    ax.set_xlabel('Num. Hidden Dense Layers')
    ax.set_ylabel('Num. Neurons Per Layer')
    ax.set_zlabel('Log10(Final Val. Loss)  (Mean Squared)')
    plt.savefig(file_prefix + "05_losses_param_space")



    with open(file_prefix + "06_statistics.txt", 'w') as f:
        f.write("Square NN")
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

if __name__ == "__main__":
    Num_Iterations = 3
    N_smallest = 1

    run_nn_tests(ScalarSquareNN, Num_Iterations, N_smallest)
    run_nn_tests(ScalarHighDNN, Num_Iterations, N_smallest)
