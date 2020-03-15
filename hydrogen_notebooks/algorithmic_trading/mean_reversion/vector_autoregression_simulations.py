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

def multivariate_normal_sample(μ, Ω, n):
    return numpy.random.multivariate_normal(μ, Ω, n)

def timeseries_plot(samples, ylabel, title, plot_name):
    nplot, nsample = samples.shape
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
        axis[i].plot(time, samples[i], lw=1.0)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def var_simulate(x0, μ, φ, Ω, n):
    m, l = x0.shape
    xt = numpy.zeros((m, n))
    ε = multivariate_normal_sample(μ, Ω, n)
    for i in range(l):
        xt[:,i] = x0[:,i]
    for i in range(l, n):
        xt[:,i] = μ + ε[i]
        for j in range(l):
            t1 = φ[j]*numpy.matrix(xt[:,i-j-1]).T
            t2 = numpy.squeeze(numpy.array(t1), axis=1)
            xt[:,i] += t2
    return xt

def phi_companion_form(φ):
    l, n, _ = φ.shape
    p = φ[0]
    for i in range(1,l):
        p = numpy.concatenate((p, φ[i]), axis=1)
    for i in range(1, n):
        if i == 1:
            r = numpy.eye(n)
        else:
            r = numpy.zeros((n, n))
        for j in range(1,l):
            if j == i - 1:
                r = numpy.concatenate((r, numpy.eye(n)), axis=1)
            else:
                r = numpy.concatenate((r, numpy.zeros((n, n))), axis=1)
        p = numpy.concatenate((p, r), axis=0)
    return p

# %%

φ = numpy.array([
        numpy.matrix([[1.0, 0.5],
                     [0.5, 1.0]]),
        numpy.matrix([[0.5, 0.3],
                     [0.2, 0.1]])
])
phi_companion_form(φ)

# %%

φ = numpy.array([
        numpy.matrix([[1.0, 0.5, 2.0],
                      [0.5, 1.0, 3.0],
                      [0.5, 1.0, 3.0]]),
        numpy.matrix([[2.0, 3.0, 4.0],
                      [7.0, 6.0, 5.0],
                      [8.0, 9.0, 10.0]]),
        numpy.matrix([[7.0, 8.0, 9.0],
                      [12.0, 11.0, 10.0],
                      [13.0, 14.0, 15.0]])
])
phi_companion_form(φ)

# %%

μ = [0.0, 0.0]
Ω = [[1.0, 0.0], [0.0, 1.0]]
φ = numpy.array([
        numpy.matrix([[1.0, 0.0],
                     [0.5, 0.0]]),
        numpy.matrix([[0.0, 0.0],
                     [0.0, 0.0]])
])
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 10
xt = var_simulate(x0, μ, φ, Ω, n)

# %%

plot_name = "var_2_simulation_1_x_y_timeseries"
title = f"VAR(2) Simulation 1"
ylabel = [r"$x$", r"$y$"]
timeseries_plot(xt, ylabel, title, plot_name)
