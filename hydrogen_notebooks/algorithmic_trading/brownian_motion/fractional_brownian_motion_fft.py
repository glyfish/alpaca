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

n = 1000
H = 0.9

# %%

C = numpy.zeros(2*n)
for i in range(2*n):
    if i == 0:
        C[i] = 1.0
    elif i < n:
        C[i] = bm.fbn_autocorrelation(H, i)
    else:
        C[i] = bm.fbn_autocorrelation(H, 2*n-i)

# %%

Λ = numpy.fft.fft(C).real
numpy.any([l < 0 for l in Λ])

# %%

dB = bm.brownian_noise(2*n)
J = numpy.zeros(2*n, dtype=numpy.cdouble)
J[0] = numpy.complex(dB[0], 0.0)
J[n] = numpy.complex(dB[n], 0.0)

for i in range(1, n):
    J[i] = numpy.sqrt(Λ[i])*numpy.complex(dB[i], dB[n+i]) / numpy.sqrt(2)
    J[2*n-i] = numpy.sqrt(Λ[2*n-i])*numpy.complex(dB[i], -dB[n+i]) / numpy.sqrt(2)

# %%
