# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
