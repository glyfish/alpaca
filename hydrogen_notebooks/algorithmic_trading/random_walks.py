# %%
%load_ext autoreload
%autoreload 2

import os
import sys
from datetime import datetime
import backtrader
from matplotlib import pyplot
from lib import config
import numpy

pyplot.style.use(config.glyfish_style)
wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')

# %%

def brownian_motion(Δt, n):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return samples

def brownian_motion_with_drift(μ, σ, Δt, n):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ * Δ * numpy.sqrt(Δt)) + (μ * Δt)
    return samples

def geometric_brownian_motion(μ, σ, s0, Δt, n):
    samples = brownian_motion_with_drift(μ, σ, Δt, n)
    return s0*numpy.exp(samples)

def plot(samples, time, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(10, 5))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.plot(time, samples)
    config.save_post_asset(figure, "mean_reversion", plot_name)

# %%

Δt = 0.01
n = 10000

samples = brownian_motion(Δt, n)
time = numpy.linspace(0.0, float(n-1)*Δt, n)
title = f"Brownian Motion; Δt={Δt}, μ={format(numpy.mean(samples), '2.2f')}, σ={format(numpy.std(samples), '2.2f')}"
plot(samples, time, title, "bownian_motion_1")
