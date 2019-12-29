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
        return σ**2*(numpy.exp(2.0*λ*t) - 1.0)/(2.0*λ)
    return f

def covariance(σ, λ, μ):
    def f(t,s):
        return σ**2*(numpy.exp(λ*(t+s)) - numpy.exp(λ*(numpy.abs(t-s))))/(2.0*λ)
    return f

def orenstein_uhlenbeck(x0, σ, λ, μ, t, nsample=1):
    μ_t = mean(x0, λ, μ)(t)
    σ_t = variance(σ, λ, μ)
    return numpy.normal.random(μ_t, σ_t)

def orenstein_uhlenbeck_series(x0, σ, λ, μ, Δt, nsample):
    samples = numpy.zeros(nsample)
    for i in range(1, nsample):
        samples[i] = orenstein_uhlenbeck(samples[i-1], σ, λ, μ, Δt)
    return samples

def time_series_plot(f, title, ylabel, plot_name):
    time = numpy.linspace(0.0, len(f)-1, len(f))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    axis.plot(time, f, lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def time_series_multiplot(series, time, λ_vals, ylabel, title, plot_name):
    nplot = len(series)
    nsample = len(series[0])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_title(title)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(ylabel)
    for i in range(nplot):
        axis.plot(time, series[i], label=f"λ={λ_vals[i]}", lw=3.0)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot_name)

# %%

nsample = 200
μ = 1.0
x0 = 1.0
tmax = 5.0
σ = 1.0
λ_vals = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25]
title = f"Ornstein-Uhlenbeck Process Mean: σ={σ}, μ={μ}"
ylabel = r"$μ(t)$"
plot_name = f"ornstein_uhlenbeck_mean_μ={μ}_x0={x0}"

time = numpy.linspace(0, tmax, nsample)
means = [mean(x0, λ_vals[i], μ)(time) for i in range(len(λ_vals))]

time_series_multiplot(means, time, λ_vals, ylabel, title, plot_name)

# %%

nsample = 200
μ = 1.0
x0 = 1.0
tmax = 5.0
σ = 1.0
λ_vals = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25]
title = f"Ornstein-Uhlenbeck Process Variance: σ={σ}, μ={μ}"
ylabel = r"$σ(t)$"
plot_name = f"ornstein_uhlenbeck_variance_μ={μ}_x0={x0}"

time = numpy.linspace(0, tmax, nsample)
vars = [variance(σ, λ_vals[i], μ)(time) for i in range(len(λ_vals))]

time_series_multiplot(vars, time, λ_vals, ylabel, title, plot_name)

# %%

nsample = 200
σ = 1.0
φ = -0.4
μ = 1.0

series = reg.ar1_series_with_offset(φ, μ, σ, nsample)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = f"ar1_example_φ={φ}_μ={μ}"
time_series_plot(series, title, r"$x_t$", plot_name)
