import numpy as np
import pickle

import keras
from keras.layers import Input, Dense, Activation, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.python.client import device_lib
import tensorflow as tf

import sklearn
import sklearn.model_selection

from numpy.random import seed


class ScalarHighDNN:

    def __init__(self, pdist_types):
        print(keras.backend.tensorflow_backend._get_available_gpus())
        print(device_lib.list_local_devices())

        self.parameters = {'num_hidden_dense': pdist_types.Linear(low=1, high=50, default=2, ptype='int'),
                           'num_neurons':      pdist_types.Linear(low=1, high=100, default=3, ptype='int'),
                           'dropout_frac':     pdist_types.Linear(low=0.05, high=0.75, default=0.1, ptype='float'),
                           'batch_size':       pdist_types.Pow2(low=2, high=128, default=16, ptype='int'),
                           'learning_rate':    pdist_types.Log10(low=0.0001, high=0.1, default=0.001, ptype='float'),
                           'epochs': pdist_types.Linear(low=10, high=500, default=100, ptype='int')
                           }

        self.model = None
        self.history = None
        self.current_params = None

    def get_prefix(self):
        return "highd_nn"         

    def get_parameter_descriptions(self):
        return self.parameters

    def prepare_model(self, params, rand_seed=0):
        self.current_params = params
        seed(rand_seed)
        tf.set_random_seed(rand_seed)
        self.model = Sequential()

        self.model.add(Dense(1, input_dim=1))

        for i in range(0, self.current_params['num_hidden_dense']-1):
            self.model.add(Dense(self.current_params['num_neurons']))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(self.current_params['dropout_frac']))
        
        self.model.add(Dense(1))

        optimizer = keras.optimizers.Adam(lr=self.current_params['learning_rate'],
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=None,
                                  decay=0.0,
                                  amsgrad=False)
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['mean_squared_error'])
        self.model.summary()


    
    def train_model(self, Xtrain, Ytrain, Xtest, Ytest, epochs=100, verbose=1, filename=None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=50)
        mc = ModelCheckpoint(filename + '.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)

        hist = self.model.fit(x=Xtrain, y=Ytrain,
                              validation_data=(Xtest, Ytest),
                              verbose=verbose,                              shuffle=True,
                              batch_size=self.current_params['batch_size'],
                              epochs=self.current_params['epochs'],
                              callbacks=[es, mc])
        self.history = hist.history
        return hist

    def predict(self, Y):
        return self.model.predict(Y)

    
    def save(self, filename):
        self.model.save(filename +'.h5')
        with open(filename + '.pk', 'wb') as f:
            pickle.dump(self.history, f)
    
    def load(self, filename, load_hist=True):
        self.model = keras.models.load_model(filename + '.h5')

        if load_hist:
            with open(filename + '.pk', 'rb') as f:
                self.history = pickle.load(f)
            return self.history
        else:
            return None