# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import special
from scipy.stats import chi2
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def scaled_brownian_noise(n):
    return numpy.random.normal(0.0, 1.0/numpy.sqrt(n), n)

def brownian_motion(bn, t):
    return sum(bn[:t])

def unit_normal(t):
    return numpy.exp(-t**2)/numpy.sqrt(2.0*numpy.pi)

def modified_chi_squared(x):
    return  2.0*numpy.exp(-(2.0*x-1.0)/2.0) / numpy.sqrt(2.0*numpy.pi*(2.0*x-1.0))

# stochastic integral simulation
# int_0^1{B(s)dB(s)}

def stochastic_integral_ensemble_1(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_1(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_1(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n):
        val += brownian_motion(bn, i-1)*bn[i]
    return val

def stochastic_integral_solution_1(n):
    return 0.5*(numpy.random.normal(0.0, 1.0, n)**2 - 1.0)

def noise_plot(samples, time, plot_name):
    nsamples = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(f"Scaled Brownian Noise, nsamples = {nsamples}")
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "regression", plot_name)

def samples_plot(pdf, samples, title, ylabel, xlabel, plot, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
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
    config.save_post_asset(figure, "regression", plot)

# %%

nsample = 100000
n = 1000
time = numpy.linspace(1.0/n, 1.0, n)

# %%

bn = scaled_brownian_noise(n)
noise_plot(bn, time, f"scaled_brownian_noise_{n}")

# %%

integral_samples = stochastic_integral_ensemble_1(n, nsample)
integral_solution_samples = stochastic_integral_solution_1(nsample)
