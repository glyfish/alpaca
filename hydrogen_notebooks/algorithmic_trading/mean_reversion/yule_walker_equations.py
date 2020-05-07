# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot
import statsmodels.api as sm
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima

pyplot.style.use(config.glyfish_style)

# %%

def yule_walker_column_vector(acf):
    return numpy.matrix(acf[1:]).T

def yule_walker_matrix(acf):
    n = len(acf) - 1
    result = numpy.matrix([acf[:n]])
    row = acf[:n]
    for i in range(1, n):
        row = numpy.roll(row, 1)
        row[0] = acf[i]
        result = numpy.concatenate((result, numpy.array([row])), axis=0)
    return result

# %%

n = 10000

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

# %%

numpy.real(arima.autocorrelation(ar1))[:2]

# %%

acf = numpy.real(arima.autocorrelation(ar2))[:3]
r = yule_walker_column_vector(acf)
R = yule_walker_matrix(acf)
numpy.linalg.inv(R)*r

# %%

acf = numpy.real(arima.autocorrelation(ar3))[:4]
r = yule_walker_column_vector(acf)
R = yule_walker_matrix(acf)
numpy.linalg.inv(R)*r

# %%

sm.regression.yule_walker(ar1, order=1, method='mle')

# %%

sm.regression.yule_walker(ar2, order=2, method='mle')

# %%

sm.regression.yule_walker(ar3, order=3, method='mle')
