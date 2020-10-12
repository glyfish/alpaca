# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot
from lib import regression as reg
from lib import stats
from lib import config
from lib import var
from lib import arima
import statsmodels.api as sm

pyplot.style.use(config.glyfish_style)

# %%

def normal(σ=1.0, μ=0.0):
    def f(x):
        ε = (x - μ)**2/(2.0*σ**2)
        return numpy.exp(-ε)/numpy.sqrt(2.0*numpy.pi*σ**2)
    return f

def brownian_motion(n):
    φ = numpy.array([1.0])
    δ = numpy.array([])
    return arima.arma_generate_sample(φ, δ, n)

# stochastic integral simulation
# \sqrt{\int_0^1{B_y^2(s)ds}}
def denominator_integral(yt):
    n = len(yt)
    y = yt-numpy.mean(yt)
    return numpy.sum(y**2)/float(n**2)

# \sqrt{\int_0^1{B_y(s)B_x(s)ds}}
def numerator_integral(xt, yt):
    n = len(yt)
    y = yt-numpy.mean(yt)
    x = xt-numpy.mean(xt)
    return numpy.sum(y*x)/float(n**2)

def beta_estimate(xt, yt):
    return numerator_integral(xt, yt) / denominator_integral(yt)

def beta_distribution_sample(npt, nsample):
    beta = numpy.zeros(nsample)
    for i in range(nsample):
        yt = brownian_motion(npt)
        xt = brownian_motion(npt)
        beta[i] = beta_estimate(yt, xt)
    return beta

def beta_ols_estimate_sample(npt, nsample):
    beta = numpy.zeros(nsample)
    for i in range(nsample):
        yt = brownian_motion(npt)
        xt = brownian_motion(npt)
        params, rsquard, err = arima.ols_estimate(xt, yt, False)
        beta[i] = params[1]
    return beta

def comparison_plot(title, samples, labels, plot):
    nplot, nsamples = samples.shape
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_xlabel(r"$t$")
    axis.set_xlim([0, nsamples-1])
    for i in range(nplot):
        axis.plot(range(nsamples), samples[i], label=labels[i], lw=1)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def regression_plot(xt, yt, params, err, β_r_squared, legend_anchor, title, plot_name, lim=None):
    nsample = len(xt)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_{t}$")
    axis.set_xlabel(r"$y_{t}$")
    if lim is not None:
        axis.set_xlim(lim)
        axis.set_ylim(lim)
        x = numpy.linspace(lim[0], lim[1], 100)
    else:
        x = numpy.linspace(numpy.min(xt), numpy.max(xt), 100)
    y_hat = x * params[1] + params[0]
    axis.set_title(title)
    axis.plot(xt, yt, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Simulation")
    axis.plot(x, y_hat, lw=3.0, color="#000000", zorder=6, label=r"$x_{t}=\hat{\alpha}+\hat{\beta}y_{t}$")
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8", alpha=0.5)
    axis.text(x[0], y_hat[80],
              r"$\hat{\beta}=$" + f"{format(params[1], '2.4f')}, " +
              r"$\sigma_{\hat{\beta}}=$" + f"{format(err[1], '2.4f')}\n"
              r"$\hat{\alpha}=$" + f"{format(params[0], '2.4f')}, " +
              r"$\sigma_{\hat{\alpha}}=$" + f"{format(err[0], '2.4f')}\n"
              r"$R^2=$"+f"{format(β_r_squared, '2.4f')}\n",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.legend(bbox_to_anchor=legend_anchor).set_zorder(7)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def distribution_comparison_plot(pdf, samples, title, plot, label=None, xrange=None, ylimit=None, bins=50, title_offset=1.0):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_β(x)$")
    axis.set_xlabel(r"β")
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
    config.save_post_asset(figure, "mean_reversion", plot)

def distribution_plot(samples, title, plot, xrange=None, ylimit=None, bins=50, title_offset=1.0):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_ylabel(r"$f_X(x)$")
    axis.set_xlabel(r"x")
    axis.set_title(title, y=title_offset)
    axis.set_prop_cycle(config.distribution_sample_cycler)
    _, bins, _ = axis.hist(samples, bins, rwidth=0.8, density=True, label=f"Samples", zorder=5)
    if xrange is None:
        delta = (bins[-1] - bins[0]) / 500.0
        xrange = numpy.arange(bins[0], bins[-1], delta)
    axis.set_xlim([xrange[0], xrange[-1]])
    if ylimit is not None:
        axis.set_ylim(ylimit)
    config.save_post_asset(figure, "mean_reversion", plot)

# %%

npt = 1000

# %%

xt = brownian_motion(npt)
yt = brownian_motion(npt)

title = f"Brownian Motion Series Comparison, n={npt}"
plot_name = "cointegration_spurious_correlation_distribution_time_series_example"
labels = [r"$x_t$", r"$y_t$"]
samples = numpy.array([xt, yt])

comparison_plot(title, samples, labels, plot_name)

# %%

params, rsquard, err = arima.ols_estimate(yt, xt)

# %%

title = f"Spurious Correlation of Independent Brownian Motion Time series, n={npt}"
plot_name = f"cointegration_spurious_correlation_distribution_spurious_correlation_regression"
regression_plot(yt, xt, params, err, rsquard, [0.7, 0.4], title, plot_name)


# %%

npt = 1000
nsample = 100000

β_estimate = beta_ols_estimate_sample(npt, nsample)

# %%

mean = numpy.mean(β_estimate)
sigma = numpy.sqrt(numpy.var(β_estimate))
title = r"OLS $\hat{\beta}$, " + f"Sample Size={nsample}, T={npt}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = "cointegration_spurious_correlation_distribution_β_estimate"
distribution_comparison_plot(normal(σ=sigma, μ=mean), β_estimate, title, plot_name, xrange=numpy.arange(-3.0, 3.1, 0.1), label="Normal PDF")

# %%

npt = 1000
nsample = 100000

β_samples = beta_distribution_sample(npt, nsample)

# %%

mean = numpy.mean(β_samples)
sigma = numpy.sqrt(numpy.var(β_samples))
title = r"$\beta=\frac{\int_{0}^{1}B_x(s)B_y(s)ds}{\int_{0}^{1}B_y^2(s)ds}$, " + f"Sample Size={nsample}, T={npt}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"cointegration_spurious_correlation_distribution_β_simulation_{npt}"
distribution_comparison_plot(normal(σ=sigma, μ=mean), β_samples, title, plot_name, xrange=numpy.arange(-3.0, 3.1, 0.1), label="Normal PDF",title_offset=1.05)

# %%

npt = 10000
nsample = 100000

β_samples = beta_distribution_sample(npt, nsample)

# %%

mean = numpy.mean(β_samples)
sigma = numpy.sqrt(numpy.var(β_samples))
title = r"$\beta=\frac{\int_{0}^{1}B_x(s)B_y(s)ds}{\int_{0}^{1}B_y^2(s)ds}$, " + f"Sample Size={nsample}, T={npt}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"cointegration_spurious_correlation_distribution_β_simulation_{npt}"
distribution_comparison_plot(normal(σ=sigma, μ=mean), β_samples, title, plot_name, xrange=numpy.arange(-3.0, 3.1, 0.1), label="Normal PDF",title_offset=1.05)
