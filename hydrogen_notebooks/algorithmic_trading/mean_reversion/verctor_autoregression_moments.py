# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from scipy import stats
from lib import config

pyplot.style.use(config.glyfish_style)

# %%

def phi_companion_form(φ):
    l, n, _ = φ.shape
    p = φ[0]
    for i in range(1,l):
        p = numpy.concatenate((p, φ[i]), axis=1)
    for i in range(1, n):
        if i == 1:
            r = numpy.eye(n)
        else:
            r = numpy.zeros((n, n))
        for j in range(1,l):
            if j == i - 1:
                r = numpy.concatenate((r, numpy.eye(n)), axis=1)
            else:
                r = numpy.concatenate((r, numpy.zeros((n, n))), axis=1)
        p = numpy.concatenate((p, r), axis=0)
    return p

# %%

φ = numpy.array([
    [[1.0, 0.5],
     [0.5, 1.0]],
    [[0.5, 0.3],
     [0.2, 0.1]]
])
phi_companion_form(φ)

# %%

φ = numpy.array([
    [[1.0, 0.5, 2.0],
     [0.5, 1.0, 3.0],
     [0.5, 1.0, 3.0]],
    [[2.0, 3.0, 4.0],
     [7.0, 6.0, 5.0],
     [8.0, 9.0, 10.0]],
    [[7.0, 8.0, 9.0],
     [12.0, 11.0, 10.0],
     [13.0, 14.0, 15.0]]
])
phi_companion_form(φ)
