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

def comparison_plot(title, samples, labels, plot):
    nplot, nsamples = samples.shape
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_xlabel(r"$t$")
    axis.set_xlim([0, nsamples-1])
    for i in range(nplot):
        axis.plot(range(nsamples), samples[i], label=labels[i], lw=1)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def cointgrated_generate_sample(φ, δ, n):
    return None

# %%

φ = numpy.array([0.8])
δ = numpy.array([])
d = 1
n = 5000

xt = arima.arima_generate_sample(φ, δ, d, n)
yt = arima.arima_generate_sample(φ, δ, d, n)
zt = xt-0.5*yt

# %%

title = r"Series Comparison for $I(1)$ Sum $\varepsilon_t=x_t-0.5y_t$, " + r"$\phi=$"+f"{numpy.array2string(φ, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ, precision=2, separator=',')}"
plot_name = "cointegated_I(1)_sum_comparison_1"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t$"]
samples = numpy.array([xt, yt, zt])

comparison_plot(title, samples, labels, plot_name)
