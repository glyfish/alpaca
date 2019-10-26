# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import special
from scipy.stats import chi2
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def scaled_brownian_noise(n):
    return numpy.random.normal(0.0, 1.0/numpy.sqrt(n), n)

def brownian_motion(bn, t):
    return sum(bn[:t])

def unit_normal(t):
    return numpy.exp(-t**2)/numpy.sqrt(2.0*numpy.pi)

def modified_chi_squared(x):
    return  2.0*numpy.exp(-(2.0*x+1.0)/2.0) / numpy.sqrt(2.0*numpy.pi*(2.0*x+1.0))

# stochastic integral simulation
# \int_0^1{B(s)dB(s)}

def stochastic_integral_ensemble_1(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_1(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_1(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n):
        val += brownian_motion(bn, i-1)*bn[i]
    return val

# Analytic Solution of integral 1
# \frac{1}{2}[B^2(1) - 1]

def stochastic_integral_solution_1(n):
    return 0.5*(numpy.random.normal(0.0, 1.0, n)**2 - 1.0)

# stochastic integral simulation
# \int_0^1{B^2(s)ds}

def stochastic_integral_ensemble_2(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_2(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_2(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return val/n

# stochastic integral simulation
# \sqrt{\int_0^1{B^2(s)ds}}

def stochastic_integral_ensemble_3(n, nsample):
    vals = numpy.zeros(nsample)
    for i in range(nsample):
        vals[i] = stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

def stochastic_integral_simulation_3(bn):
    n = len(bn)
    val = 0.0
    for i in range(1, n+1):
        val += brownian_motion(bn, i-1)**2
    return numpy.sqrt(val/n)

# Dickey-Fuller Statisti distribution
# \frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_0^1{B^2(s)ds}}

def dickey_fuller_test_statistic_ensemble(n, nsample):
    vals = numpy.zeros(nsample)
    numerator = stochastic_integral_solution_1(nsample)
    for i in range(nsample):
        vals[i] = numerator[i] / stochastic_integral_simulation_3(scaled_brownian_noise(n))
    return vals

# Plots

def noise_plot(samples, time, plot_name):
    nsamples = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(f"Scaled Brownian Noise, nsamples = {nsamples}")
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "regression", plot_name)

def distribution_comparison_plot(pdf, samples, title, plot, label=None, xrange=None, ylimit=None, bins=50, title_offset=1.0):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_X(x)$")
    axis.set_xlabel(r"x")
    axis.set_title(title, y=title_offset)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, bins, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    sample_distribution = [pdf(val) for val in xrange]
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    if label is None:
        label=f"Target PDF"
    axis.plot(xrange, sample_distribution, label=label, zorder=6)
    axis.legend(bbox_to_anchor=(0.75, 0.9))
    config.save_post_asset(figure, "regression", plot)

def distribution_plot(samples, title, plot, xrange=None, ylimit=None):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_X(x)$")
    axis.set_xlabel(r"x")
    axis.set_title(title)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, 50, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    config.save_post_asset(figure, "regression", plot)

# %%

n = 1000
time = numpy.linspace(1.0/n, 1.0, n)

# %%

bn = scaled_brownian_noise(n)
noise_plot(bn, time, f"scaled_brownian_noise_{n}")

# %%

nsample = 10000
integral_samples = stochastic_integral_ensemble_1(n, nsample)
integral_solution_samples = stochastic_integral_solution_1(nsample)

# %%

mean = numpy.mean(integral_solution_samples)
sigma = numpy.sqrt(numpy.var(integral_solution_samples))
title = r"$\frac{1}{2}[B^2(1) - 1]$, " + f"Sample Size={nsample}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_1_solution_{nsample}"
distribution_comparison_plot(modified_chi_squared, integral_solution_samples, title, plot_name, xrange=numpy.arange(-0.43, 3.0, 0.01))

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\int_{0}^{1}B(s)dB(s)$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_1_{nsample}"
distribution_comparison_plot(modified_chi_squared, integral_samples, title, plot_name, xrange=numpy.arange(-0.43, 3.0, 0.01))

# %%

nsample = 1000
integral_samples = stochastic_integral_ensemble_1(n, nsample)
integral_solution_samples = stochastic_integral_solution_1(nsample)

# %%

mean = numpy.mean(integral_solution_samples)
sigma = numpy.sqrt(numpy.var(integral_solution_samples))
title = r"$\frac{1}{2}[B^2(1) - 1]$, " + f"Sample Size={nsample}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_1_solution_{nsample}"
distribution_comparison_plot(modified_chi_squared, integral_solution_samples, title, plot_name, xrange=numpy.arange(-0.45, 3.0, 0.01))

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\int_{0}^{1}B(s)dB(s)$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_1_{nsample}"
distribution_comparison_plot(modified_chi_squared, integral_samples, title, plot_name, xrange=numpy.arange(-0.45, 3.0, 0.01))

# %%

n = 1000
nsample = 10000
integral_samples = stochastic_integral_ensemble_2(n, nsample)

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\int_{0}^{1}B^2(s)ds$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_2_{nsample}"
distribution_plot(integral_samples, title, plot_name, xrange=numpy.arange(0.0, 4.0, 0.01))

# %%

n = 1000
nsample = 1000
integral_samples = stochastic_integral_ensemble_2(n, nsample)

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\int_{0}^{1}B^2(s)ds$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_2_{nsample}"
distribution_plot(integral_samples, title, plot_name, xrange=numpy.arange(0.0, 4.0, 0.01))

# %%

n = 1000
nsample = 10000
integral_samples = stochastic_integral_ensemble_3(n, nsample)
numpy.mean(integral_samples)

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\sqrt{\int_{0}^{1}B^2(s)ds}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_3_{nsample}"
distribution_plot(integral_samples, title, plot_name, xrange=numpy.arange(0.0, 3.0, 0.01))

# %%

n = 1000
nsample = 1000
integral_samples = stochastic_integral_ensemble_3(n, nsample)
numpy.mean(integral_samples)

# %%

mean = numpy.mean(integral_samples)
sigma = numpy.sqrt(numpy.var(integral_samples))
title = r"$\sqrt{\int_{0}^{1}B^2(s)ds}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"stochastic_integral_simulation_3_{nsample}"
distribution_plot(integral_samples, title, plot_name, xrange=numpy.arange(0.0, 3.0, 0.01))

# %%

n = 1000
nsample = 1000
test_statistic_samples = dickey_fuller_test_statistic_ensemble(n, nsample)

# %%

mean = numpy.mean(test_statistic_samples)
sigma = numpy.sqrt(numpy.var(test_statistic_samples))
title = r"t=$\frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"dickey_fuller_distribution_simulation_{nsample}"
distribution_comparison_plot(unit_normal, test_statistic_samples, title, plot_name, xrange=numpy.arange(-4.0, 8.0, 0.01), label="Unit Normal", bins=50, title_offset=1.05)

# %%

n = 1000
nsample = 10000
test_statistic_samples = dickey_fuller_test_statistic_ensemble(n, nsample)

# %%

mean = numpy.mean(test_statistic_samples)
sigma = numpy.sqrt(numpy.var(test_statistic_samples))
title = r"t=$\frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"dickey_fuller_distribution_simulation_{nsample}"
distribution_comparison_plot(unit_normal, test_statistic_samples, title, plot_name, xrange=numpy.arange(-4.0, 8.0, 0.01), label="Unit Normal", bins=100, title_offset=1.05)

# %%

n = 1000
nsample = 100000
test_statistic_samples = dickey_fuller_test_statistic_ensemble(n, nsample)

# %%

mean = numpy.mean(test_statistic_samples)
sigma = numpy.sqrt(numpy.var(test_statistic_samples))
title = r"t=$\frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"dickey_fuller_distribution_simulation_{nsample}"
distribution_comparison_plot(unit_normal, test_statistic_samples, title, plot_name, xrange=numpy.arange(-4.0, 8.0, 0.01), label="Unit Normal", bins=200, title_offset=1.05)
