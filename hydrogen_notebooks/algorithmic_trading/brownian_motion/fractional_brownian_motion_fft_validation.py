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

f = [8.0, 4.0, 8.0, 0.0]
t = numpy.fft.fft(f)
numpy.fft.ifft(t)

# %% FFT algorithm for FBN example

# circulant matrix is 1 at n=0 and zero everywhere else
c = numpy.zeros(10)
c[0] = 1

numpy.fft.fft (c).real
