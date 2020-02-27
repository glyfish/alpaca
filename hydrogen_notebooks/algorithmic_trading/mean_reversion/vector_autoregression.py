# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion
from lib import regression as reg
from lib import adf
from statsmodels.tsa import stattools

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
