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

n = 10000

# ##

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
plot_name = "acf_pacf_ar_comparison"
arima.timeseries_comparison_plot(samples, params, 500, title, plot_name)

# %%

θ1 = numpy.array([0.2])
ma1 = arima.ma_generate_sample(θ1, n)

θ2 = numpy.array([0.2, 0.3])
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
