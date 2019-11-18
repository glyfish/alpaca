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

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def ar1_parameter_plot(series, φ_hat, φ_hat_var, φ_r_squared, legend_anchor, title, plot_name, lim=None):
    nsample = len(series)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_{t}$")
    axis.set_xlabel(r"$x_{t-1}$")
    if lim is not None:
        axis.set_xlim(lim)
        axis.set_ylim(lim)
        x = numpy.linspace(lim[0], lim[1], 100)
    else:
        x = numpy.linspace(numpy.min(series), numpy.max(series), 100)
    y_hat = x * φ_hat
    axis.set_title(title)
    axis.plot(series[1:], series[0:-1], marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Simulation")
    axis.plot(x, y_hat, lw=3.0, color="#000000", zorder=6, label=r"$x_{t}=\hat{\phi}x_{t-1}$")
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    axis.text(x[80], x[0],
              r"$\hat{\phi}=$" + f"{format(φ_hat, '2.3f')}\n" +
              r"$\sigma_{\hat{\phi}}=$" + f"{format(numpy.sqrt(φ_hat_var), '2.3f')}\n"
              r"$R^2=$"+f"{format(φ_r_squared, '2.3f')}\n",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.legend(bbox_to_anchor=legend_anchor).set_zorder(7)
    config.save_post_asset(figure, "regression", plot_name)

def φ_estimate(series):
    cov = numpy.sum(series[1:]*series[0:-1])
    var = numpy.sum(series[0:-1]**2)
    return cov/var

def φ_estimate_var(series, var=1.0):
    return var/numpy.sum(series[0:-1]**2)

def φ_r_squared(series, φ):
    y = series[1:]
    x = series[0:-1]
    y_bar = numpy.mean(y)
    ε = y - φ*x
    ssr = numpy.sum(ε**2)
    sst = numpy.sum((y-y_bar)**2)
    return 1.0 - ssr/sst

# %%

nsample = 1000
σ = 1.0
φ = 0.5
series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.2, 0.8], title, plot_name)

# %%
# y = φx + c

x = series[0:-1]
y = series[1:]
A = numpy.vstack([x,numpy.ones(len(x))]).T

a, residuals, rank, b = numpy.linalg.lstsq(A, y, rcond=None)
a
residuals
numpy.sum((0.5*x-y)**2)

# %%

nsample = 1000
σ = 1.0
φ = -0.5

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.7, 0.9], title, plot_name)

# %%

nsample = 1000
σ = 1.0
φ = 0.9

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.25, 0.9], title, plot_name)

# %%

nsample = 1000
σ = 1.0
φ = -0.9

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.85, 0.85], title, plot_name)

# %%

nsample = 1000
σ = 1.0
φ = 0.1

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.25, 0.8], title, plot_name)

# %%

nsample = 1000
σ = 1.0
φ = 0.99

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.5, 0.85], title, plot_name)

# %%

nsample = 1000
σ = 1.0
φ = 1.01

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = φ_estimate(series)
φ_hat_var = φ_estimate_var(series)
r_squared = φ_r_squared(series, φ)

# %%

title = f"AR(1) Series: σ={σ}, φ={φ}"
plot_name = f"ar1_parameter_estimation_σ_{σ}_φ_{φ}"
ar1_parameter_plot(series, φ_hat, φ_hat_var, r_squared, [0.5, 0.85], title, plot_name, lim=[-11.0, 10.0])
