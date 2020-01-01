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
    σ_t = numpy.sqrt(variance(σ, λ, μ)(t))
    return numpy.random.normal(μ_t, σ_t, nsample)

def orenstein_uhlenbeck_series(x0, σ, λ, μ, Δt, nsample):
    samples = numpy.zeros(nsample)
    for i in range(1, nsample):
        samples[i] = orenstein_uhlenbeck(samples[i-1], σ, λ, μ, Δt)
    return samples

def orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample):
    samples = numpy.zeros(nsample)
    for i in range(1, nsample):
        dxt = λ*samples[i-1]*Δt + μ*Δt + σ*numpy.sqrt(Δt)*numpy.random.normal(0.0, 1.0)
        samples[i] = samples[i-1] + dxt
    return samples

def plot(f, x, title, xlabel, ylabel, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(ylabel)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.plot(x, f, lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def time_series_plot(samples, time, text_pos, title, ylabel, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples), '2.2f')}\nσ={format(numpy.var(samples), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    axis.set_ylabel(ylabel)
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def multiplot(series, time, λ_vals, xlabel, ylabel, title, plot_name):
    nplot = len(series)
    nsample = len(series[0])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    for i in range(nplot):
        axis.plot(time, series[i], label=f"λ={λ_vals[i]}", lw=3.0)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ar1_ensemble(φ, μ, σ, tmax, nsample):
    samples = numpy.zeros(nsample)
    for i in range(nsample):
        samples[i] = reg.ar1_series_with_offset(φ, μ, σ, tmax)[-1]
    return samples

# %%

nsample = 200
μ = 1.0
x0 = 1.0
tmax = 5.0
σ = 1.0
λ_vals = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25]
title = f"Ornstein-Uhlenbeck Process Mean: σ={σ}, μ={μ}"
ylabel = r"$μ(t)$"
xlabel = r"$t$"
plot_name = f"ornstein_uhlenbeck_mean_μ={μ}_x0={x0}"

time = numpy.linspace(0, tmax, nsample)
means = [mean(x0, λ_vals[i], μ)(time) for i in range(len(λ_vals))]

multiplot(means, time, λ_vals, xlabel, ylabel, title, plot_name)

# %%

nsample = 200
μ = 1.0
x0 = 1.0
tmax = 5.0
σ = 1.0
λ_vals = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25]
title = f"Ornstein-Uhlenbeck Process Variance: σ={σ}, μ={μ}"
ylabel = r"$σ(t)$"
xlabel = r"$t$"
plot_name = f"ornstein_uhlenbeck_variance_μ={μ}_x0={x0}"

time = numpy.linspace(0, tmax, nsample)
vars = [variance(σ, λ_vals[i], μ)(time) for i in range(len(λ_vals))]

multiplot(vars, time, λ_vals, xlabel, ylabel, title, plot_name)

# %%

title = f"Ornstein-Uhlenbeck Half-Life of Mean Decay"
ylabel = r"$\frac{\ln{2}}{\lambda}$"
xlabel = r"$\lambda$"
plot_name = f"ornstein_uhlenbeck_variance_half_life_of_mean_decay"

λ = numpy.linspace(-2.0, -0.1, 100)
halflife = [-numpy.log(2)/i for i in λ]

plot(halflife, λ, title, xlabel, ylabel, plot_name)

# %%
# Verify ornstein-uhlenbeck implementaion and compare with ar1 simulation

φ = -0.4
λ = φ - 1
μ = 1.0
x0 = 0.0
σ = 1.0

tmax = 100
nsample = 100000

ar1_samples = ar1_ensemble(φ, μ, σ, tmax, nsample)
print(f"ΑR(1) μ={numpy.mean(ar1_samples)}")
print(f"ΑR(1) σ={numpy.var(ar1_samples)}")

oh_samples = orenstein_uhlenbeck(x0, σ, λ, μ, tmax, nsample)
print(f"OH μ={numpy.mean(oh_samples)}")
print(f"OH σ={numpy.var(oh_samples)}")

# %%

nsample = 200
σ = 1.0
φ = -0.4
μ = 1.0

tmax = 200
time = numpy.linspace(0, tmax, nsample)
series = reg.ar1_series_with_offset(φ, μ, σ, nsample)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = f"ornstein_uhlenbeck_ar1_example_φ={φ}_μ={μ}"
text_pos = [150.0, 2.0]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 1.0
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: λ={λ}, σ={σ}, μ={μ}, σ={σ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_difference_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [150.0, 2.0]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.9
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: λ={λ}, σ={σ}, μ={μ}, σ={σ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_difference_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [125.0, 2.0]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.5
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: λ={λ}, σ={σ}, μ={μ}, σ={σ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_difference_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [75.0, 1.75]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.1
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: λ={λ}, σ={σ}, μ={μ}, σ={σ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_difference_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [15.0, 1.5]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.01
nsample = 2000
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_difference_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: λ={λ}, σ={σ}, μ={μ}, σ={σ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_difference_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [15.0, 1.5]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 1.0
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Simulated Series: φ={φ}, σ={σ}, μ={μ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [150.0, 1.5]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.1
nsample = 200
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Simulated Series: φ={φ}, σ={σ}, μ={μ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [15.0, 1.5]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)

# %%

φ = -0.4
λ = φ - 1.0
μ = 1.0
x0 = 0.0
σ = 1.0

Δt = 0.01
nsample = 2000
time = numpy.linspace(0, Δt*(nsample-1), nsample)

series = orenstein_uhlenbeck_series(x0, σ, λ, μ, Δt, nsample)

title = f"Ornstein-Uhlenbeck Difference Series: φ={φ}, σ={σ}, μ={μ}, Δt={Δt}"
plot_name = f"ornstein_uhlenbeck_simulation_λ={λ}_μ={μ}_σ={σ}_Δt={Δt}"
text_pos = [15.0, 1.5]
time_series_plot(series, time, text_pos, title, r"$x_t$", plot_name)
