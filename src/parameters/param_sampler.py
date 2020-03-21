import math
import random

class ParamSampler:
    def __init__(self):
        pass
    
    def sample_parameters(self, param_desc, method='default'):
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

