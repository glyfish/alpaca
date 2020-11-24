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

n = 10000

# %%

φ1 = numpy.array([0.7])
ar1 = arima.ar_generate_sample(φ1, n)

φ2 = numpy.array([0.5, 0.3])
ar2 = arima.ar_generate_sample(φ2, n)

φ3 = numpy.array([0.2, 0.3, 0.4])
ar3 = arima.ar_generate_sample(φ3, n)

# %%

samples = numpy.array([ar1, ar2, ar3])
params = [r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"]
title = "AR(p) Comparison"
plot_name = "acf_pacf_ar_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)

# %%

title = f"AR(1) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_1_pacf_acf_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar1, ylim, max_lag, plot_name)

# %%

title = f"AR(2) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_2_pacf_acf_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar2, ylim, max_lag, plot_name)

# %%

title = f"AR(3) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_3_pacf_acf_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar3, ylim, max_lag, plot_name)

# %%

φ1 = numpy.array([0.7])
ar1 = arima.ar_generate_sample(φ1, n)

φ2 = numpy.array([0.0, 0.3])
ar2 = arima.ar_generate_sample(φ2, n)

φ3 = numpy.array([0.0, 0.0, 0.4])
ar3 = arima.ar_generate_sample(φ3, n)

# %%

samples = numpy.array([ar1, ar2, ar3])
params = [r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}",
          r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"]
title = "AR(p) Comparison"
plot_name = "acf_pacf_ar_single_lag_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)

# %%

title = f"AR(1) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_1_pacf_acf_single_lag_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar1, ylim, max_lag, plot_name)

# %%

title = f"AR(2) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_2_pacf_acf_single_lag_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar2, ylim, max_lag, plot_name)

# %%

title = f"AR(3) ACF-PACF Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}"
plot_name = "acf_pacf_ar_3_pacf_acf_single_lag_comparison"
max_lag = 10
ylim = [-0.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ar3, ylim, max_lag, plot_name)

# %%

θ1 = numpy.array([0.7])
ma1 = arima.ma_generate_sample(θ1, n)

θ2 = numpy.array([0.5, 0.3])
ma2 = arima.ma_generate_sample(θ2, n)

θ3 = numpy.array([0.2, 0.3, 0.4])
ma3 = arima.ma_generate_sample(θ3, n)

# %%

samples = numpy.array([ma1, ma2, ma3])
params = [r"$\theta=$"+f"{numpy.array2string(θ1, precision=2, separator=',')}",
          r"$\theta=$"+f"{numpy.array2string(θ2, precision=2, separator=',')}",
          r"$\theta=$"+f"{numpy.array2string(θ3, precision=2, separator=',')}"]
title = "MA(q) Comparison"
plot_name = "acf_pacf_ma_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)

# %%

title = f"MA(1) ACF-PACF Comparison: " + r"$\theta=$"+f"{numpy.array2string(θ1, precision=2, separator=',')}"
plot_name = "acf_pacf_ma_1_pacf_acf_comparison"
max_lag = 10
ylim = [-1.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ma1, ylim, max_lag, plot_name)

# %%

title = f"MA(2) ACF-PACF Comparison: " + r"$\theta=$"+f"{numpy.array2string(θ2, precision=2, separator=',')}"
plot_name = "acf_pacf_ma_2_pacf_acf_comparison"
max_lag = 10
ylim = [-1.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ma2, ylim, max_lag, plot_name)

# %%

title = f"MA(3) ACF-PACF Comparison: " + r"$\theta=$"+f"{numpy.array2string(θ3, precision=2, separator=',')}"
plot_name = "acf_pacf_ma_3_pacf_acf_comparison"
max_lag = 10
ylim = [-1.1, 1.1]
arima.acf_yule_walker_pcf_plot(title, ma3, ylim, max_lag, plot_name)
