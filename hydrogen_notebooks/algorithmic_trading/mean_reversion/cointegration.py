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

def cointgrated_generate_sample(φ, δ, n):
    return None

def corrletaion_plot(xt, yt, φ_hat, φ_hat_var, φ_r_squared, legend_anchor, title, plot_name, lim=None):
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

def ols_correlation_estimate(xt, yt):
    model = sm.OLS(yt, xt)
    results = model.fit()
    print(results.summary())
    return results.params, results.rsquared, results.bse

# %%

φ = numpy.array([0.8])
δ = numpy.array([])
d = 1
n = 5000

xt = arima.arima_generate_sample(φ, δ, d, n)
yt = arima.arima_generate_sample(φ, δ, d, n)
zt = xt-0.5*yt

# %%

title = r"Series Comparison for $I(1)$ Sum $\varepsilon_t=x_t-0.5y_t$, " + r"$\phi=$"+f"{numpy.array2string(φ, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ, precision=2, separator=',')}"
plot_name = "cointegated_I(1)_sum_comparison_1"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t$"]
samples = numpy.array([xt, yt, zt])

comparison_plot(title, samples, labels, plot_name)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%
