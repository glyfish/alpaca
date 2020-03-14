# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from scipy import stats
from lib import config

pyplot.style.use(config.glyfish_style)

# %%

def timeseries_plot(samples, ylabel, title, plot_name):
    nsample, nplot = samples.shape
    ymin = numpy.amin(samples)
    ymax = numpy.amax(samples)
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        axis[i].set_ylabel(ylabel[i])
        axis[i].set_ylim([ymin, ymax])
        axis[i].set_xlim([0.0, nsample])
        axis[i].plot(time, samples[:,i], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def var_simulate(x0, μ, φ, Ω, n):
    m, l = x0.shape
    xt = numpy.zeros((m, n))
    for i in range(l):
        xt[:,i] = x0[:,i]
    return xt

# %%
