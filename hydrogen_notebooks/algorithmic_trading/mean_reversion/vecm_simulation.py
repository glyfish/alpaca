# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima
import statsmodels.api as sm
import scipy

pyplot.style.use(config.glyfish_style)

# %%
