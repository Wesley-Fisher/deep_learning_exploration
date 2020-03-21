import math
import time
import random
import numpy as np
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor

class ParamSampler:
    def __init__(self):
        pass
    
    def sample_parameters_basic(self, param_desc, method='default'):
        params = {}

        if method == 'default':
            sampler = self.sample_default
        elif method == 'random':
            sampler = self.sample_random
        else:
            raise KeyError('No method key [%s]' % method)

        for param in param_desc:
            x = sampler(param_desc[param])
            params[param] = x
        return params
    
    def sample_default(self, description):
        return description[2]
    
    def sample_random(self, description):
        low = description[0]
        high = description[1]
        vtype = description[3]
        dist = description[4]

        def sample_between(low, high):
            return random.uniform(low, high)

        if dist == 'lin':
            low = low
            high = high
            x = sample_between(low, high)
            x = x
        elif dist == 'log10':
            low = math.log10(low)
            high = math.log10(high)
            x = sample_between(low, high)
            x = math.pow(10, x)
        else:
            raise KeyError('Unknown dist key [%s]' % dist)
        
        if vtype == 'int':
            return int(x)
        elif vtype == 'float':
            return float(x)
        else:
            raise KeyError('Unknown vtype key [%s]' % vtype)

    def sample_parameters_gp(self, description, history, key):
        if len(history.history) == 0:
            return self.sample_parameters(description, 'default')
        
        input_params = history.in_params
        X, Y = self.create_gp_data(history, input_params, key)
        
        # Based on maximization
        Y = -np.log(Y)

        gpr = gpr = GaussianProcessRegressor()
        gpr.fit(X, Y)

        fxp = np.max(Y)

        # Set some defaults
        EI_max = 0
        xd_max = self.sample_parameters_basic(description, 'default')

        for i in range(0, 100*len(input_params)):
            xd = self.sample_parameters_basic(description, 'random')
            x = []
            for p in input_params:
                x.append(xd[p])
            x = np.array(x)

            EI = self.calculate_EI(gpr, x, fxp)

            if EI > EI_max:
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


    def create_gp_data(self, history, params, key):
        
        xs = []
        ys = []

        for h in history.history:
            x = []
            for p in params:
                x.append(h[p])

            x = np.array(x)
            xs.append(x)

            y = h[key]
            ys.append(y)

        X = np.stack(xs, axis=0)
        Y = np.array(ys)
        return X, Y

    def sample_parameters_gp_explore(self, description, history, key):

        # One some change, just take random parameters
        random.seed(time.time)
        if random.randint(0, 100) > 90:
            return self.sample_parameters_basic(description, 'random')
        
        return self.sample_parameters_gp(description, history, key)

        