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

def generate_ensemble(arima_params, ecm_params, n, m):
    samples  = numpy.array(arima.ecm_sample_generate(arima_params, ecm_params, n))
    for i in range(m-1):
        samples = numpy.append(samples, numpy.array(arima.ecm_sample_generate(arima_params, ecm_params, n)), axis=0)
    return samples

def ensemble_plot(samples, text_pos, title, ylab, plot_name):
    nsim, npts = samples.shape
    figure, axis = pyplot.subplots(figsize=(12, 8))
    time = numpy.linspace(0, npts-1, npts)
    axis.set_xlabel("Time")
    axis.set_ylabel(ylab)
    axis.set_title(title)
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples[:,-1]), '2.2f')}\nσ={format(numpy.std(samples[:,-1]), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    for i in range(nsim):
        axis.plot(time, samples[i], lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ensemble_mean_plot(mean, time, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, mean)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ensemble_std_plot(H, std, time, lengend_location, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    step = int(len(time) / 10)
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, std, label="Ensemble Average")
    axis.plot(time[::step], time[::step]**H, label=r"$t^{H}$", marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    axis.legend(bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ensemble_autocorrelation_plot(H, ac, time, lengend_location, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    step = int(len(time) / 10)
    axis.set_ylim([-1.0, 1.0])
    axis.set_xlabel("Time")
    axis.set_title(title)
    label = r"$\frac{1}{2}[(t-1)^{2H} + (t+1)^{2H} - 2t^{2H})]$"
    axis.plot(time, ac, label="Ensemble Average")
    axis.plot(time[1::step], bm.fbn_autocorrelation(H, time[1::step]), label=label, marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    axis.legend(bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "mean_reversion", plot_name)

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

def var_xt(φ, n):
    sum = 0.0
    for k in range(1, n):
        sum += 2.0*(n-k)*φ**k
    return (n+sum) / (1-φ**2)

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
m = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

samples = generate_ensemble(arima_params, ecm_params, n, m)

# %%

σ = numpy.sqrt(var_xt(arima_params['φ'][0], n))

title = f"ECM Ensemple, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, " + r"$\hat{\sigma}=$" + f"{format(σ, '2.2f')}, size={m}"
plot_name = f"ecm_properies_ensemble_x_t{image_postfix}"
ylab = r"$x_t$"
ensemble_plot(samples[0::2], [10.0, 100.0], title, ylab, plot_name)

# %%

title = f"ECM Ensemple, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properies_ensemble_x_t{image_postfix}"
ylab = r"$y_t$"
ensemble_plot(samples[1::2], [10.0, 75.0], title, ylab, plot_name)
