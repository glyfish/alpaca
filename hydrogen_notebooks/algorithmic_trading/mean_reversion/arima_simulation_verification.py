# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
import scipy
from matplotlib import pyplot
import statsmodels.api as sm
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima

pyplot.style.use(config.glyfish_style)

# %%

def arima_generate_sample(samples, d):
    n = len(samples)
    if d == 1:
        return numpy.cumsum(samples)
    else:
        result = numpy.zeros(n)
        result[0], result[1] = samples[0], samples[1]
        for i in range(2, n):
            result[i] = samples[i] + 2.0*result[i-1] - result[i-2]
        return result

def normal_pdf(μ, σ):
    def f(x):
        return scipy.stats.norm.pdf(x, μ, σ)
    return f

def pdf_samples(title, pdf, samples, plot, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_xlabel(r"$x$")
    axis.set_title(title)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, 50, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    axis.plot(xrange, sample_distribution, label=f"Target PDF", zorder=6)
    axis.legend(bbox_to_anchor=(0.75, 0.9))
    config.save_post_asset(figure, "mean_reversion", plot)

def timeseries_comparison_plot(samples, tmax, title, ylables, plot_name):
    nplot, nsample = samples.shape
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, tmax-1, tmax)
    for i in range(nplot):
        stats=f"μ={format(numpy.mean(samples[i]), '2.2f')}\nσ={format(numpy.std(samples[i]), '2.2f')}"
        bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
        axis[i].text(0.05, 0.75, stats, fontsize=15, bbox=bbox, transform=axis[i].transAxes)
        axis[i].set_ylabel(ylables[i])
        ymin = 1.1*numpy.amin(samples[i,:tmax])
        ymax = 1.1*numpy.amax(samples[i,:tmax])
        axis[i].set_ylim([ymin, ymax])
        axis[i].set_xlim([0.0, tmax])
        axis[i].plot(time, samples[i,:tmax], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

# %%
# ARIMA(1,1,0) example

φ1 = numpy.array([0.8])
δ1 = numpy.array([])
d = 1
n = 10000

arma1 = arima.arma_generate_sample(φ1, δ1, n)
arima.adf_report(arma1)

# %%

arima1 = arima_generate_sample(arma1, d)
darima11 = arima.sample_difference(arima1)

# %%
arima.adf_report(darima11)

# %%
arima.adf_report(darima11)

# %%

samples = numpy.array([arima1[1:], darima11])
title = r"ARIMA(1,1,0) $\Delta x_t$-ARMA(1,0) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ1, precision=2, separator=',')}"
plot_name = "aima_comparison_1_1_0"
ylables = [r"$x_t$", r"$\Delta x_t$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)

# %%
# ARIMA(1,2,0) example

φ1 = numpy.array([0.8])
δ1 = numpy.array([])
d = 2
n = 10000

arma2 = arima.arma_generate_sample(φ1, δ1, n)

# %%

arima.adf_report(arma2)

# %%

arima2 = arima_generate_sample(arma2, d)
darima21 = arima.sample_difference(arima2)
darima22 = arima.sample_difference(darima21)

# %%

arima.adf_report(arima2)

# %%

arima.adf_report(darima21)

# %%

arima.adf_report(darima22)

# %%

samples = numpy.array([arma2[2:], darima22])
title = r"ARIMA(1,1,0) $\Delta x_t^2$-ARMA(1,0) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ1, precision=2, separator=',')}"
plot_name = "aima_comparison_1_1_0"
ylables = [r"$x_t$", r"$\Delta x_t^2$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)
