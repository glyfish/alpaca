# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import stats
from lib import config
from lib import var

pyplot.style.use(config.glyfish_style)

# %%

def yt_parameter_estimation_form(xt1, xt2):
    m, n = xt1.shape
    ones = numpy.ones((1, n))
    return numpy.concatenate((xt1, xt2), axis=0)

def theta_parameter_estimation_form(xt):
    l, n = xt.shape
    xt1 = xt[:,1:n-1]
    xt2 = xt[:,:n-2]
    yt = yt_parameter_estimation_form(xt1, xt2)
    m, _ = yt.shape
    yy = numpy.matrix(numpy.zeros((m, m)))
    xy = numpy.matrix(numpy.zeros((l, m)))
    for i in range(l, n):
        x = numpy.matrix(xt[:,i]).T
        y = numpy.matrix(yt[:,i-l]).T
        yy += y*y.T
        xy += x*y.T
    return xy*numpy.linalg.inv(yy)

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.2],
                      [0.2, 0.3]]),
        numpy.matrix([[0.3, 0.0],
                     [0.0, 0.4]])
])
var.eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 1000
xt = var.var_simulate(x0, μ, φ, ω, n)

# %%

M = var.stationary_mean(φ, μ)
Σ = var.stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_estimation_1_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
var.timeseries_plot(xt, ylabel, title, plot_name)

# %%

n = 50000
xt = var.var_simulate(x0, μ, φ, ω, n)
theta = theta_parameter_estimation_form(xt)
numpy.matrix(theta[:,0:2])
numpy.matrix(theta[:,2:])

# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.2],
                      [0.2, 0.3]]),
        numpy.matrix([[0.3, 0.2],
                     [0.1, 0.4]])
])
var.eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 1000
xt = var.var_simulate(x0, μ, φ, ω, n)

# %%

M = var.stationary_mean(φ, μ)
Σ = var.stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_estimation_2_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
var.timeseries_plot(xt, ylabel, title, plot_name)

# %%

n = 50000
xt = var.var_simulate(x0, μ, φ, ω, n)
theta = theta_parameter_estimation_form(xt)
numpy.matrix(theta[:,0:2])
numpy.matrix(theta[:,2:])
# %%

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, -0.2],
                      [0.2, 0.3]]),
        numpy.matrix([[0.3, -0.2],
                     [0.1, 0.4]])
])
var.eigen_values(φ)
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 1000
xt = var.var_simulate(x0, μ, φ, ω, n)

# %%

M = var.stationary_mean(φ, μ)
Σ = var.stationary_covariance_matrix(φ, ω)
cov = stats.covaraince(xt[0], xt[1])
plot_name = "var_2_estimation_1_x_y_timeseries"
title = f"VAR(2) Simulation: γ={format(Σ[0,1], '2.2f')}, " + \
         r"$\hat{\gamma}$=" + f"{format(cov, '2.2f')}, " + \
         r"$μ_x$=" + f"{format(M[0,0], '2.2f')}, " + \
         r"$σ_x$=" + f"{format(numpy.sqrt(Σ[0,0]), '2.2f')}, " + \
         r"$μ_y$=" + f"{format(M[1,0], '2.2f')}, " + \
         r"$σ_y$=" + f"{format(numpy.sqrt(Σ[1,1]), '2.2f')}"
ylabel = [r"$x$", r"$y$"]
var.timeseries_plot(xt, ylabel, title, plot_name)

# %%

n = 50000
xt = var.var_simulate(x0, μ, φ, ω, n)
theta = theta_parameter_estimation_form(xt)
numpy.matrix(theta[:,0:2])
numpy.matrix(theta[:,2:])
