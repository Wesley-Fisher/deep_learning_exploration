import math
import time
import random
import numpy as np
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor

class ParamSampler:
    def __init__(self, param_desc, history):
        self.param_desc = param_desc
        self.history = history
    
    def sample_parameters_default(self):
        params = {}
        for p in self.param_desc.keys():
            x, xn = self.param_desc[p].sample_default()
            params[p] = x
        return params

    def sample_parameters_random(self):
        params = {}
        for p in self.param_desc.keys():
            x, xn = self.param_desc[p].sample_random()
            params[p] = x
        return params

    def sample_parameters_gp_explore(self, key):

        # One some chance, just take random parameters
        random.seed(time.time)
        if random.randint(0, 100) > 90:
            return self.sample_parameters_random()
        
        return self.sample_parameters_gp(key)
    
    def sample_parameters_gp(self, key):
        if len(self.history.history) == 0:
            return self.sample_parameters_default()
        
        in_param_ordering = self.history.in_params
        X, Y = self.create_gp_data( in_param_ordering, key)
        
        # Based on maximization
        Y = -np.log(Y)

        gpr = gpr = GaussianProcessRegressor()
        gpr.fit(X, Y)

        fxp = np.max(Y)

        # Set some defaults
        EI_max = 0
        xd_max = None

        for i in range(0, 100*len(in_param_ordering)):
            x = [] # Values to sample
            xd = {} # Dict of params to output
            
            for p in in_param_ordering:
                xa, xn = self.param_desc[p].sample_random()
                x.append(xn)
                xd[p] = xa
            x = np.array(x)

            EI = self.calculate_EI(gpr, x, fxp)

            if i == 0 or EI > EI_max:
                EI_max = EI
                xd_max = xd

        return xd_max
            
    
    def calculate_EI(self, gpr, x, fxp):
        x = np.reshape(x, (1, -1)) # Single sample

        u, dev = gpr.predict(x, return_std=True)
        zeta = 0.01

        if dev == 0:
            EI = 0
        else:
            Z = (u - fxp - zeta) / dev
            PHI = norm.cdf(Z)
            phi = norm.pdf(Z)
            EI = (u - fxp - zeta)*PHI + dev*phi
        return EI


    def create_gp_data(self, param_keys_ordered, key):
        
        xs = []
        ys = []

        for h in self.history.history:
            x = []
            for p in param_keys_ordered:
                xi = h[p]
                xn = self.param_desc[p].normalize(xi)

                x.append(xn)
            
            x = np.array(x)
            xs.append(x)

            y = h[key]
            ys.append(y)

        X = np.stack(xs, axis=0)
        Y = np.array(ys)
        return X, Y
    
    def sample_parameters_lowest(self, key):
        h_best = self.history.history[0]
        score_min = h_best[key]

        for hist in history.history:
            score = hist[key]

            if score <= score_min:
                score_min = score
                h_best = hist
        
        return h_best




        