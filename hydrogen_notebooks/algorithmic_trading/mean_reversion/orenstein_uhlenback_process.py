# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion
from lib import regression as reg
from lib import adf
from statsmodels.tsa import stattools

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def mean(x0, λ, μ):
    def f(t):
        return x0*numpy.exp(λ*t) + μ*(numpy.exp(λ*t)-1.0)/λ
    return f

def variance(σ, λ, μ):
    def f(t):
        return σ**2(numpy.exp(2.0*λ*t) - 1.0)/(2.0*λ)
    return f

def covariance(σ, λ, μ):
    def f(t,s):
        return σ**2*(numpy.exp(λ*(t+s)) - numpy.exp(λ*(numpy.abs(t-s))))/(2.0*λ)
    return f

def plot(samples, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"t")
    axis.set_title(title)
    axis.plot(t, samples, lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

def time_series_plot(f, title, plot_name):
    time = numpy.linspace(0.0, len(f)-1, len(f))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    axis.plot(time, f, lw=1)
    config.save_post_asset(figure, "regression", plot_name)

# %%

nsample = 1000
σ = 1.0
φ = 0.5
μ = 1.0

series = reg.ar1_series_with_offset(φ, μ, σ, nsample)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = f"adf_example_φ={φ}_μ={μ}"
adf_time_series_plot(series, title, plot_name)
