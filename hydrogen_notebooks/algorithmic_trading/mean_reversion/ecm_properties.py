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

def corrletation_plot(xt, yt, params, err, β_r_squared, legend_anchor, title, plot_name, lim=None):
    nsample = len(xt)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$y_{t}$")
    axis.set_xlabel(r"$x_{t}$")
    if lim is not None:
        axis.set_xlim(lim)
        axis.set_ylim(lim)
        x = numpy.linspace(lim[0], lim[1], 100)
    else:
        x = numpy.linspace(numpy.min(xt), numpy.max(xt), 100)
    y_hat = x * params[1] + params[0]
    axis.set_title(title)
    axis.plot(xt, yt, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Simulation")
    axis.plot(x, y_hat, lw=3.0, color="#000000", zorder=6, label=r"$y_{t}=\hat{\alpha}+\hat{\beta}x_{t}$")
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

# %%
