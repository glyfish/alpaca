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

def brownian_noise(σ, n):
    return numpy.random.normal(0.0, σ, n)

def arq_series(q, φ, σ, n):
    samples = numpy.zeros(n)
    ε = brownian_noise(σ, n)
    for i in range(q, n):
        samples[i] = ε[i]
        for j in range(0, q):
            samples[i] += φ[j] * samples[i-(j+1)]
    return samples

def ar1_var(φ, σ):
    return σ**2/(1.0-φ**2)

def ar1_auto_correlation(n, φ, σ):
    return φ**2*σ**2

def ar1_comparission_multiplot(series, φ, ylim, title, plot_name):
    nplot = len(series)
    nsample = len(series[0])
    figure, axis = pyplot.subplots(3, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    time = numpy.linspace(0, nsample-1, nsample)
    for i in range(nplot):
        axis[i].set_ylabel(r"$x_t$")
        axis[i].set_ylim(ylim)
        axis[i].set_xlim([0.0, nsample])
        axis[i].text(time[int(0.9*nsample)], 0.65*ylim[-1], f"φ={φ[i]}", fontsize=16)
        axis[i].plot(time, series[i], lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

def ar1_auto_correlation_plot(series, φ, nplot, title, plot_name, ylim):
    ac = reg.autocorrelate(series)
    ac_eq = [φ**n for n in range(nplot)]
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$γ_{\tau}$")
    axis.set_xlabel(r"Time Lag $(\tau)$")
    if ylim is None:
        axis.set_ylim([-0.1, 1.0])
    else:
        axis.set_ylim(ylim)
    axis.set_title(title)
    axis.plot(range(nplot), numpy.real(ac[:nplot]), marker='o', markersize=10.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, label="Simulation", zorder=6)
    axis.plot(range(nplot), ac_eq, lw="2", label=r"$φ^{\tau}$", zorder=5)
    config.save_post_asset(figure, "regression", plot_name)

# %%

nseries = 3
nsample = 500
σ = 1.0

φ = numpy.array([0.1, 0.5, 0.9])

series = []
for i in range(0, nseries):
    series.append(arq_series(1, [φ[i]], σ, nsample))

# %%

title = f"AR(1) Series Comparison: σ={σ}"
plot_name = "ar1_equilibrium_series_comparison_1"
ar1_comparission_multiplot(series, φ, [-8.0, 8.0], title, plot_name)

# %%

nseries = 3
nsample = 500
σ = 1.0

φ = numpy.array([-0.1, -0.5, -0.9])

series = []
for i in range(0, nseries):
    series.append(arq_series(1, [φ[i]], σ, nsample))

# %%

title = f"AR(1) Series Comparison: σ={σ}"
plot_name = "ar1_equilibrium_series_comparison_2"
ar1_comparission_multiplot(series, φ, [-8.0, 8.0], title, plot_name)

# %%

nsample = 100000
σ = 1.0
φ = 0.1
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Μean: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_μ_φ_{format(φ, '2.1f')}"
reg.cumulative_mean_plot(series, 0.0, title, plot, ylim=[-0.55, 0.25], legend_pos=[0.85, 0.9])

# %%

title = f"AR(1) Cumulative Variance: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_var_φ_{format(φ, '2.1f')}"
reg.cumulative_var_plot(series, var, title, plot, ylim=[0.55, 1.2], legend_pos=[0.85, 0.9])

# %%

nsample = 100000
σ = 1.0
φ = 0.5
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Μean: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_μ_φ_{format(φ, '2.1f')}"
reg.cumulative_mean_plot(series, 0.0, title, plot, ylim=[-0.15, 0.6], legend_pos=[0.85, 0.9])

# %%

title = f"AR(1) Cumulative Variance: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_var_φ_{format(φ, '2.1f')}"
reg.cumulative_var_plot(series, var, title, plot, ylim=[0.55, 1.4], legend_pos=[0.75, 0.7])

# %%

nsample = 100000
σ = 1.0
φ = 0.9
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Μean: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_μ_φ_{format(φ, '2.1f')}"
reg.cumulative_mean_plot(series, 0.0, title, plot, ylim=[-0.8, 1.8], legend_pos=[0.85, 0.9])

# %%

title = f"AR(1) Cumulative Variance: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_var_φ_{format(φ, '2.1f')}"
reg.cumulative_var_plot(series, var, title, plot, ylim=[0.5, 6.5], legend_pos=[0.85, 0.5])

# %%

nsample = 100000
σ = 1.0
φ = -0.5
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Μean: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_μ_φ_{format(φ, '2.1f')}"
reg.cumulative_mean_plot(series, 0.0, title, plot, ylim=[-0.2, 0.2], legend_pos=[0.85, 0.9])

# %%

title = f"AR(1) Cumulative Variance: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_cumulative_var_φ_{format(φ, '2.1f')}"
reg.cumulative_var_plot(series, var, title, plot, ylim=[0.9, 2.75], legend_pos=[0.85, 0.9])


# %%

nsample = 10000
σ = 1.0
φ = 0.9
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Autocorrelation: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_autocorelation_φ_{format(φ, '2.1f')}"
ar1_auto_correlation_plot(series, φ, 100, title, plot, ylim=[-0.1, 1.1])

# %%

nsample = 10000
σ = 1.0
φ = -0.9
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Autocorrelation: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_autocorelation_φ_{format(φ, '2.1f')}"
ar1_auto_correlation_plot(series, φ, 100, title, plot, ylim=[-1.1, 1.1])

# %%

nsample = 10000
σ = 1.0
φ = 0.5
var = ar1_var(φ, σ)
series = arq_series(1, [φ], σ, nsample)

# %%

title = f"AR(1) Cumulative Autocorrelation: σ={σ}, φ={format(φ, '2.1f')}"
plot = f"ar1_equilibrium_series_autocorelation_φ_{format(φ, '2.1f')}"
ar1_auto_correlation_plot(series, φ, 30, title, plot, ylim=[-0.1, 1.1])
