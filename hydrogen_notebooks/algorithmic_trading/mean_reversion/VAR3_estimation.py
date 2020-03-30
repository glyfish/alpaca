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

μ = [0.0, 0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.0, 0.0],
                      [0.0, 0.3, 0.0],
                      [0.0, 0.0, 0.4]]),
        numpy.matrix([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]]),
        numpy.matrix([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]])
])
var.eigen_values(φ)
x0 = numpy.array([[0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0]])
n = 10000
xt = var.var_simulate(x0, μ, φ, ω, n)

# %%

M = var.stationary_mean(φ, μ)
Σ = var.stationary_covariance_matrix(φ, ω)
plot_name = "var_3_estimation_1_x_y_timeseries"
title = f"VAR(3) Simulation"
ylabel = [r"$x$", r"$y$", r"$z$"]
var.timeseries_plot(xt, 1000, ylabel, title, plot_name)

# %%

Σ[:3,:3]

# %%

stats.covariance(xt[0], xt[1])

# %%

stats.covariance(xt[1], xt[2])

# %%

stats.covariance(xt[0], xt[2])

# %%

theta = var.theta_parameter_estimation(xt)
var.split_theta(theta)

# %%

var.omega_parameter_estimation(xt, theta)

# %%

l = 50
Σt = var.stationary_autocovariance_matrix(φ, ω, l)

# %%

title = f"VAR(3) Simulation x(t) Autocorrelation"
plot_name = "var_3_simulation_1_x_autocorrelation"
γt = [Σt[i, 0, 0] for i in range(l)] / Σt[0, 0, 0]

var.autocorrelation_plot(title, xt[0], γt, [-0.05, 1.05], plot_name)

# %%

title = f"VAR(3) Simulation y(t) Autocorrelation"
plot_name = "var_3_simulation_1_y_autocorrelation"
γt = [Σt[i, 1, 1] for i in range(l)] / Σt[0, 1, 1]

var.autocorrelation_plot(title, xt[1], γt, [-0.05, 1.05], plot_name)

# %%

title = f"VAR(3) Simulation z(t) Autocorrelation"
plot_name = "var_3_simulation_1_z_autocorrelation"
γt = [Σt[i, 2, 2] for i in range(l)] / Σt[0, 2, 2]

var.autocorrelation_plot(title, xt[2], γt, [-0.05, 1.05], plot_name)

# %%

μ = [0.0, 0.0, 0.0]
ω = numpy.matrix([[1.0, 0.5, -0.5],
                  [0.5, 1.0, 0.0],
                  [-0.5, 0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.2, -0.4],
                      [0.0, 0.3, 0.0],
                      [0.2, 0.0, 0.4]]),
        numpy.matrix([[0.1, 0.0, 0.2],
                      [0.0, 0.25, 0.0],
                      [0.0, 0.2, -0.3]]),
        numpy.matrix([[0.4, 0.0, 0.25],
                      [0.0, 0.1, 0.0],
                      [0.25, 0.0, 0.2]])
])
var.eigen_values(φ)
x0 = numpy.array([[0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0]])
n = 10000
xt = var.var_simulate(x0, μ, φ, ω, n)

# %%

M = var.stationary_mean(φ, μ)
Σ = var.stationary_covariance_matrix(φ, ω)
plot_name = "var_3_estimation_2_x_y_timeseries"
title = f"VAR(3) Simulation"
ylabel = [r"$x$", r"$y$", r"$z$"]
var.timeseries_plot(xt, 1000, ylabel, title, plot_name)

# %%

Σ[:3,:3]

# %%

stats.covariance(xt[0], xt[1])

# %%

stats.covariance(xt[1], xt[2])

# %%

stats.covariance(xt[0], xt[2])

# %%

theta = var.theta_parameter_estimation(xt)
var.split_theta(theta)

# %%

var.omega_parameter_estimation(xt, theta)

# %%

l = 50
Σt = var.stationary_autocovariance_matrix(φ, ω, l)

# %%

title = f"VAR(3) Simulation x(t) Autocorrelation"
plot_name = "var_3_simulation_2_x_autocorrelation"
γt = [Σt[i, 0, 0] for i in range(l)] / Σt[0, 0, 0]

var.autocorrelation_plot(title, xt[0], γt, [-0.05, 1.05], plot_name)

# %%

title = f"VAR(3) Simulation y(t) Autocorrelation"
plot_name = "var_3_simulation_2_y_autocorrelation"
γt = [Σt[i, 1, 1] for i in range(l)] / Σt[0, 1, 1]

var.autocorrelation_plot(title, xt[1], γt, [-0.05, 1.05], plot_name)

# %%

title = f"VAR(3) Simulation z(t) Autocorrelation"
plot_name = "var_3_simulation_2_z_autocorrelation"
γt = [Σt[i, 2, 2] for i in range(l)] / Σt[0, 2, 2]

var.autocorrelation_plot(title, xt[2], γt, [-0.05, 1.05], plot_name)

# %%

title = f"VAR(2) Simulation x(t)y(t) Cross Correlation"
plot_name = "var_3_simulation_2_xy_autocorrelation"
γt = [Σt[i, 0, 1] for i in range(l)]

var.cross_correlation_plot(title, xt[0], xt[1], γt, [-0.05, 1.0], plot_name)

# %%

title = f"VAR(2) Simulation y(t)x(t) Cross Correlation"
plot_name = "var_3_simulation_2_yx_autocorrelation"
γt = [Σt[i, 1, 0] for i in range(l)]

var.cross_correlation_plot(title, xt[1], xt[0], γt, [-0.05, 1.0], plot_name)

# %%

title = f"VAR(2) Simulation x(t)z(t) Cross Correlation"
plot_name = "var_3_simulation_2_xz_autocorrelation"
γt = [Σt[i, 0, 2] for i in range(l)]

var.cross_correlation_plot(title, xt[0], xt[2], γt, [-0.1, 1.1], plot_name)

# %%

title = f"VAR(2) Simulation x(t)z(t) Cross Correlation"
plot_name = "var_3_simulation_2_zx_autocorrelation"
γt = [Σt[i, 2, 0] for i in range(l)]

var.cross_correlation_plot(title, xt[2], xt[0], γt, [-0.1, 0.8], plot_name)

# %%

title = f"VAR(2) Simulation y(t)z(t) Cross Correlation"
plot_name = "var_3_simulation_2_yz_autocorrelation"
γt = [Σt[i, 1, 2] for i in range(l)]

var.cross_correlation_plot(title, xt[1], xt[2], γt, [-0.05, 0.8], plot_name)

# %%

title = f"VAR(2) Simulation z(t)y(t) Cross Correlation"
plot_name = "var_3_simulation_2_zy_autocorrelation"
γt = [Σt[i, 2, 1] for i in range(l)]

var.cross_correlation_plot(title, xt[2], xt[1], γt, [-0.05, 0.5], plot_name)
