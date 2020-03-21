import math
import time
import random
import numpy as np
from scipy.stats import norm


class BaseDistribution:
    def __init__(self, low=None, high=None, default=None, ptype=None):
        self.low = low
        self.high = high
        self.default = default
        self.type = ptype

    def to_type(self, a):
        if self.type == 'int':
            return int(a)
        elif self.type == 'float':
            return float(a)
        else:
            raise TypeError('Unknown type [%s]' % self.type)
    
    def sample_default(self):
        x = self.default
        x = self.to_type(x)
        xn = self.normalize(x)
        return x, xn

    def sample_random(self):
        l = self.transform_to_distribution(self.low)
        h = self.transform_to_distribution(self.high)
        x = random.uniform(l, h)
        x = self.transform_from_distribution(x)
        x = self.to_type(x)

        xn = self.normalize(x)
        return x, xn

    def normalize(self, x):
        l = self.transform_to_distribution(self.low)
        h = self.transform_to_distribution(self.high)
        x = self.transform_to_distribution(x)
        
        return (x - l) / (h - l)

    def transform_to_distribution(self, a):
        raise NotImplementedError
    
    def transform_from_distribution(self, a):
        raise NotImplementedError



class LinearDistribution(BaseDistribution):
    def transform_to_distribution(self, a):
        return a
    def transform_from_distribution(self, a):
        return a

class Log10Distribution(BaseDistribution):
    def transform_to_distribution(self, a):
        return math.log10(a)
    def transform_from_distribution(sef, a):
        return math.pow(10, a)

class DistrbutionTypes:
    Linear = LinearDistribution
    Log10 = Log10Distribution