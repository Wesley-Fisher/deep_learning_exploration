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

from utils.directories import Directories as DIR
from parameters.param_distributions import DistrbutionTypes

file_prefix = DIR.RESULTS + "/distributions/"
N = 50
s = [i for i in range(0, 50)]


#
# Linear
#
dist = DistrbutionTypes.Linear(low=1, high=100, default=10, ptype='float')
xlin = []
for i in range(0, N):
    xlin.append(dist.sample_random()[0])
xlin.sort()

fig = plt.figure(1)
plt.scatter(s, xlin)
plt.title("Linear Distribution Sampling")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.savefig(file_prefix + "01_linear")
plt.close()

#
# Log10
#
dist = DistrbutionTypes.Log10(low=1, high=100, default=10, ptype='float')
xlin = []
for i in range(0, N):
    xlin.append(dist.sample_random()[0])
xlin.sort()

fig = plt.figure(2)
plt.scatter(s, xlin)
plt.title("Log10 Distribution Sampling")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.savefig(file_prefix + "02_log10")
plt.close()


#
# Pow2
#
dist = DistrbutionTypes.Pow2(low=2, high=32, default=8, ptype='int')
xlin = []
for i in range(0, N):
    xlin.append(dist.sample_random()[0])
xlin.sort()

fig = plt.figure(2)
plt.scatter(s, xlin)
plt.title("Pow2 Distribution Sampling")
plt.xlabel("Samples")
plt.ylabel("Value")
plt.savefig(file_prefix + "03_pow2")
plt.close()