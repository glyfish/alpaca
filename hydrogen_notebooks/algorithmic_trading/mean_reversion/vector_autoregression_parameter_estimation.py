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

μ = [0.0, 0.0]
ω = numpy.matrix([[1.0, 0.0], [0.0, 1.0]])
φ = numpy.array([
        numpy.matrix([[0.2, 0.1],
                      [0.1, 0.3]]),
        numpy.matrix([[0.2, 0.25],
                     [0.25, 0.3]])
])
x0 = numpy.array([[0.0, 1.0], [0.0, 1.0]])
n = 5000
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
