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

def time_series_plot(f, t, φ_hat, φ_hat_var, φ_r_squared, title, plot_name):
    time = numpy.linspace(0.0, len(f)-1, len(f))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    axis.plot(time, f, lw=1)
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    axis.text(0.8*max(time), 0.1*(max(f) - min(f)) + min(f),
              r"$t=$" + f"{format(t, '3.3f')}\n" +
              r"$\hat{\phi}=$" + f"{format(φ_hat, '2.3f')}\n" +
              r"$\sigma_{\hat{\phi}}=$" + f"{format(numpy.sqrt(φ_hat_var), '2.3f')}\n"
              r"$R^2=$"+f"{format(φ_r_squared, '2.3f')}\n",
              bbox=bbox, fontsize=14.0, zorder=7)
    config.save_post_asset(figure, "regression", plot_name)

def adf_time_series_plot(f, title, plot_name):
    time = numpy.linspace(0.0, len(f)-1, len(f))
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"$t$")
    axis.set_title(title)
    axis.plot(time, f, lw=1)
    config.save_post_asset(figure, "regression", plot_name)

def df_test(series):
    adf_result = stattools.adfuller(series, regression='nc')
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))

def adf_test(series):
    adf_result = stattools.adfuller(series, regression='c')
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))

def adf_test_with_trend(series):
    adf_result = stattools.adfuller(series, regression='ct')
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    for key, value in adf_result[4].items():
	       print('\t%s: %.3f' % (key, value))

# %%

n = 1000
nsample = 10000
test_statistic_samples = adf.df_test_statistic_ensemble(n, nsample)

# %%

mean = numpy.mean(test_statistic_samples)
sigma = numpy.sqrt(numpy.var(test_statistic_samples))
pdf, t = adf.pdf_histogram(test_statistic_samples, [-4.0, 8.0])
title = r"t=$\frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
ylabel = r"$f_T(t)$"
plot_name = f"df_pdf_histogram_{nsample}"
adf.histogram_plot(t[:-1], pdf, title, ylabel, plot_name, title_offset=1.05)

# %%

cdf = adf.cdf_histogram(t, pdf)
title = r"t=$\frac{\frac{1}{2}[B^2(1) - 1]}{\sqrt{\int_{0}^{1}B^2(s)ds}}$, " + f"Sample Size={nsample}, T={n}, μ={format(mean, '1.2f')}, σ={format(sigma, '1.2f')}"
plot_name = f"df_cdf_histogram_{nsample}"
ylabel = r"$F_T(t)$"
adf.histogram_plot(t[:-1], cdf, title, ylabel, plot_name, title_offset=1.05)

# %%
# Dickey-Fuller test example
# Fit to AR(1) with no constant: Δx_t = δx_(t-1) + ε_t

nsample = 1000
σ = 1.0
φ = 0.5

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = reg.φ_estimate(series)
φ_hat_var = reg.φ_estimate_var(series)
r_squared = reg.φ_r_squared(series, φ)
t = adf.adf_statistic(series)

title = f"AR(1) Series: φ={φ}, σ={σ}"
plot_name = "adf_example_1"
time_series_plot(series, t, φ_hat, φ_hat_var, r_squared, title, plot_name)

# %%

df_test(series)

# %%

nsample = 1000
σ = 1.0
φ = 0.99

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = reg.φ_estimate(series)
φ_hat_var = reg.φ_estimate_var(series)
r_squared = reg.φ_r_squared(series, φ)
t = adf.adf_statistic(series)

title = f"AR(1) Series: φ={φ}, σ={σ}"
plot_name = "adf_example_2"
time_series_plot(series, t, φ_hat, φ_hat_var, r_squared, title, plot_name)

# %%

df_test(series)

# %%

nsample = 500
σ = 1.0
φ = 1.01

series = reg.arq_series(1, [φ], σ, nsample)
φ_hat = reg.φ_estimate(series)
φ_hat_var = reg.φ_estimate_var(series)
r_squared = reg.φ_r_squared(series, φ)
t = adf.adf_statistic(series)

title = f"AR(1) Series: φ={φ}, σ={σ}"
plot_name = "adf_example_3"
time_series_plot(series, t, φ_hat, φ_hat_var, r_squared, title, plot_name)

# %%

df_test(series)

# %%

nsample = 1000
σ = 1.0
φ = 0.5
μ = 1.0

series = reg.ar1_series_with_offset(φ, μ, σ, nsample)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = "adf_example_with_mean_1"
adf_time_series_plot(series, title, plot_name)

# %%

adf_test(series)

# %%

nsample = 1000
σ = 1.0
φ = 0.99
μ = 1.0

series = reg.ar1_series_with_offset(φ, μ, σ, nsample)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = "adf_example_with_mean_1"
adf_time_series_plot(series, title, plot_name)

# %%

adf_test(series)

# %%

nsample = 1000
σ = 1.0
φ = 1.01
μ = 1.0

series = reg.ar1_series_with_offset(φ, μ, σ, nsample)
φ_hat = reg.φ_estimate(series)
φ_hat_var = reg.φ_estimate_var(series)
r_squared = reg.φ_r_squared(series, φ)
t = adf.adf_statistic(series)

title = f"AR(1) Series with constant offset: φ={φ}, σ={σ}, μ={μ}"
plot_name = "adf_example_with_mean_1"
adf_time_series_plot(series, title, plot_name)

# %%

adf_test(series)
