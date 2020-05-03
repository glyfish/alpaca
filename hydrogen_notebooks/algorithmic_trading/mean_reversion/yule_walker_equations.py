# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot
from statsmodels.tsa.api import VAR as pyvar
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima

pyplot.style.use(config.glyfish_style)

# %%

def yule_walker_column_vector(acf, p):
    return acf

# %%

n = 5000

φ1 = numpy.array([0.2])
ar1 = arima.ar_generate_sample(φ1, n)

φ2 = numpy.array([0.2, 0.3])
ar2 = arima.ar_generate_sample(φ2, n)

φ3 = numpy.array([0.2, 0.3, 0.4])
ar3 = arima.ar_generate_sample(φ3, n)

# %%

samples = numpy.array([ar1, ar2, ar3])
params = [r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"]
title = "AR(p) Comparison"
plot_name = "yule_walker_equations_ar_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)
