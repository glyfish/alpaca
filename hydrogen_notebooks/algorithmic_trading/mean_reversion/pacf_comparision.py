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

pyplot.style.use(config.glyfish_style)

# %%

def pcf_comparison_plot(title, samples, ylim, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    pacf = arima.pacf(samples, max_lag)
    yw_pacf = arima.yule_walker(samples, max_lag)
    axis.set_title(title)
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([-0.1, max_lag+1])
    axis.set_ylim(ylim)
    axis.plot(range(max_lag+1), pacf, label="statsmodels")
    axis.plot(range(1, max_lag+1), yw_pacf, label="Yule-Walker", marker='o', markersize=15.0, linestyle="None", markeredgewidth=1.0)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

# %%

n = 10000

# ##

φ1 = numpy.array([0.7])
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
plot_name = "pacf_method_comparison_ar_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)

# %%

title = f"AR(1) PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}"
plot_name = "pacf_method_comparison_ar_1_yule_walker_pacf"
max_lag = 10
ylim = [-0.1, 1.1]
pcf_comparison_plot(title, ar1, ylim, max_lag, plot_name)

# %%

title = f"AR(2) PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}"
plot_name = "pacf_method_comparison_ar_2_yule_walker_pacf"
max_lag = 10
ylim = [-0.1, 1.1]
pcf_comparison_plot(title, ar2, ylim, max_lag, plot_name)

# %%

title = f"AR(3) PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"
plot_name = "pacf_method_comparison_ar_3_yule_walker_pacf"
max_lag = 10
ylim = [-0.1, 1.1]
pcf_comparison_plot(title, ar3, ylim, max_lag, plot_name)
