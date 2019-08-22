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
