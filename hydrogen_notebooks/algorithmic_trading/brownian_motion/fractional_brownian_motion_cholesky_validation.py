# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm

wd = os.getcwd()

yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
# example autocorrelation calculation verification

H = 0.9
n = 3

# Autocorrelation matrix
R = numpy.matrix(numpy.array([[1.0, 0.741, 0.63, 0.58],
                              [0.741, 1.0, 0.741, 0.63],
                              [0.63, 0.741, 1.0, 0.741],
                              [0.58, 0.63, 0.741, 1.0]]))
bm.fbn_autocorrelation(H, 1)
bm.fbn_autocorrelation(H, 2)
bm.fbn_autocorrelation(H, 3)

# Test calculation of autocorrelation matrix
R_ = bm.fbn_autocorrelation_matrix(H, n)
R_

# Test caclculation of cholesky decomposition implementation
l = bm.cholesky_decompose(H, n)
l

# verify that result gives autocorrelation matrix
l*l.T

# Comparison with numpy impelementaion
c = numpy.linalg.cholesky(R_)
c

# verify that result gives autocorrelation matrix
c*c.T

# %%

H = 0.5
n = 100

# Test calculation of autocorrelation matrix
R = bm.fbn_autocorrelation_matrix(H, n)
R

# Comparison with numpy impelementaion
c = numpy.linalg.cholesky(R)
c
