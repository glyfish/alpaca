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
from statsmodels.tsa.arima_model import ARIMA as pyarima

pyplot.style.use(config.glyfish_style)

# %%

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

φ1 = numpy.array([0.8])
δ1 = numpy.array([])
d = 1
n = 10000

arima1 = arima.arima_generate_sample(φ1, δ1, d, n)
darima1 = arima.sample_difference(arima1)

# %%

samples = numpy.array([arima1[:n-1], darima1])
title = "ARIMA(1,1,0) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ1, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ1, precision=2, separator=',')}"
plot_name = "aima_estimation_1_1_0"
ylables = [r"$x_t$", r"$\Delta x_t$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)

# %%

model = pyarima(arima1, order=(1, 1, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%

title = "ARIMA(1,1,0) Residual Distribution Comparison with Normal(0, 1)"
plot_name = "aima_estimation_1_1_0_residual_distribution"
pdf_samples(title, normal_pdf(0.0, 1.0), model_fit.resid, plot_name, xrange=None, ylimit=None)

# %%

φ2 = numpy.array([0.8])
δ2 = numpy.array([])
d = 2
n = 10000

arima1 = arima.arima_generate_sample(φ2, δ2, d, n)
darima1 = arima.sample_difference(arima1)
darima2 = arima.sample_difference(darima1)

# %%

samples = numpy.array([arima1[:n-2], darima1[:n-2], darima2])
title = "ARIMA(1,2,0) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ2, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ2, precision=2, separator=',')}"
plot_name = "aima_estimation_1_2_0"
ylables = [r"$x_t$", r"$\Delta x_t$", r"$\Delta^2 x_t$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)

# %%

model = pyarima(arima1, order=(1, 2, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%

title = "ARIMA(1,2,0) Residual Distribution Comparison with Normal(0, 1)"
plot_name = "aima_estimation_1_2_0_residual_distribution"
pdf_samples(title, normal_pdf(0.0, 1.0), model_fit.resid, plot_name, xrange=None, ylimit=None)

# %%

φ3 = numpy.array([0.4, -0.3])
δ3 = numpy.array([])
d = 1
n = 10000

arima1 = arima.arima_generate_sample(φ3, δ3, d, n)
darima1 = arima.sample_difference(arima1)

# %%

samples = numpy.array([arima1[:n-1], darima1])
title = "ARIMA(2,1,0) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ3, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ3, precision=2, separator=',')}"
plot_name = "aima_estimation_2_1_0"
ylables = [r"$x_t$", r"$\Delta x_t$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)

# %%

model = pyarima(arima1, order=(2, 1, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%

title = "ARIMA(2,1,0) Residual Distribution Comparison with Normal(0, 1)"
plot_name = "aima_estimation_2_1_0_residual_distribution"
pdf_samples(title, normal_pdf(0.0, 1.0), model_fit.resid, plot_name, xrange=None, ylimit=None)

# %%

φ4 = numpy.array([])
δ4 = numpy.array([0.4, -0.3])
d = 1
n = 10000

arima1 = arima.arima_generate_sample(φ4, δ4, d, n)
darima1 = arima.sample_difference(arima1)

# %%

samples = numpy.array([arima1[:n-1], darima1])
title = "ARIMA(0,1,2) Comparison: " + r"$\phi=$"+f"{numpy.array2string(φ4, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ4, precision=2, separator=',')}"
plot_name = "aima_estimation_0_1_2"
ylables = [r"$x_t$", r"$\Delta x_t$"]
timeseries_comparison_plot(samples, 500, title, ylables, plot_name)

# %%

model = pyarima(arima1, order=(0, 1, 2))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# %%

title = "ARIMA(0,1,2) Residual Distribution Comparison with Normal(0, 1)"
plot_name = "aima_estimation_0_1_2_residual_distribution"
pdf_samples(title, normal_pdf(0.0, 1.0), model_fit.resid, plot_name, xrange=None, ylimit=None)
