import numpy as np
import pickle

import keras
from keras.layers import Input, Dense, Activation
from keras.models import Sequential

from tensorflow.python.client import device_lib

import sklearn
import sklearn.model_selection

from numpy.random import seed
from tensorflow import set_random_seed


class ScalarSquareNN:

    def __init__(self, pdist_types):
        print(keras.backend.tensorflow_backend._get_available_gpus())
        print(device_lib.list_local_devices())

        self.parameters = {'num_hidden_dense': pdist_types.Linear(low=1, high=50, default=2, ptype='int'),
                           'num_neurons':      pdist_types.Linear(low=1, high=100, default=3, ptype='int'),
                           }

        self.model = None
        self.history = None

    def get_prefix(self):
        return "square_nn"        

    def get_parameter_descriptions(self):
        return self.parameters

    def prepare_model(self, params, rand_seed=0):
        seed(rand_seed)
        set_random_seed(rand_seed)
        self.model = Sequential()

        self.model.add(Dense(1, input_dim=1))

        for i in range(0, params['num_hidden_dense']-1):
            self.model.add(Dense(params['num_neurons']))
            self.model.add(Activation('relu'))
        
        self.model.add(Dense(1))

        optimizer = keras.optimizers.Adam(lr=0.001,
                                  beta_1=0.9,
                                  beta_2=0.999,
                                  epsilon=None,
                                  decay=0.0,
                                  amsgrad=False)
        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['mean_squared_error'])
        self.model.summary()
    
    def train_model(self, Xtrain, Ytrain, Xtest, Ytest, epochs=100, verbose=1):
        hist = self.model.fit(x=Xtrain, y=Ytrain,
                              validation_data=(Xtest, Ytest),
                              verbose=verbose, callbacks=None,  shuffle=True,
                              batch_size=32, epochs=epochs)
        self.history = hist.history
        return hist

    def predict(self, Y):
        return self.model.predict(Y)
    
    def save(self, filename):
        self.model.save(filename +'.h5')
        with open(filename + '.pk', 'wb') as f:
            pickle.dump(self.history, f)

    def load(self, filename):
        self.model = keras.models.load_model(+ filename + '.h5')

        with open(filename + '.pk', 'rb') as f:
            self.history = pickle.load(f)
        return self.history
