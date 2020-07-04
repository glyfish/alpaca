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
    axis.set_title(title, pad=10)
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples[:,-1]), '2.2f')}\nσ={format(numpy.std(samples[:,-1]), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    for i in range(nsim):
        axis.plot(time, samples[i], lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ensemble_average_plot(mean, title, ylab, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    npts = len(mean)
    time = numpy.linspace(0, npts-1, npts)
    axis.set_xlabel("Time")
    axis.set_ylabel(ylab)
    axis.set_title(title, pad=10)
    axis.plot(time, mean)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def ensemble_comparison_plot(ensemble_prop, ensemble_time, stationary_prop, stationary_time, title, ylab, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel(ylab)
    axis.set_title(title, pad=10)
    axis.plot(ensemble_time, ensemble_prop, label="Ensemble Average")
    axis.plot(stationary_time, stationary_prop, label="Stationary", marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    axis.legend()
    config.save_post_asset(figure, "mean_reversion", plot_name)

def comparison_plot(title, samples, labels, plot):
    nplot, nsamples = samples.shape
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title, pad=10)
    axis.set_xlabel(r"$t$")
    axis.set_xlim([0, nsamples-1])
    for i in range(nplot):
        axis.plot(range(nsamples), samples[i], label=labels[i], lw=1)
    axis.legend(fontsize=16)
    config.save_post_asset(figure, "mean_reversion", plot)

def var_xt(φ, n):
    sum = 0.0
    for k in range(1, n):
        sum += 2.0*(n-k)*φ**k
    return (n+sum) / (1-φ**2)

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
m = 500
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

samples = generate_ensemble(arima_params, ecm_params, n, m)

# %%

σ = numpy.sqrt(var_xt(arima_params['φ'][0], n))
xt = samples[0::2]

title = f"ECM Ensemble, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, " + r"$\hat{\sigma}=$" + f"{format(σ, '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_x_t{image_postfix}"
ylab = r"$x_t$"
ensemble_plot(xt, [10.0, 100.0], title, ylab, plot_name)

# %%

title = f"ECM Ensemble, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_x_t{image_postfix}"
ylab = r"$y_t$"
yt = samples[1::2]
ensemble_plot(yt, [10.0, 60.0], title, ylab, plot_name)

# %%

β_estimate = numpy.zeros(m)
for i in range(m):
    params, rsquard, err = arima.ols_estimate(xt[i], yt[i], False)
    β_estimate[i] = params[1]

β_avg = numpy.mean(β_estimate)
β_std = numpy.std(β_estimate)

# %%

εt = yt - β_avg*xt

title = f"ECM Ensemble, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_ε_t{image_postfix}"
ylab = r"$\varepsilon_t$"
ensemble_plot(εt, [10.0, 3.0], title, ylab, plot_name)

# %%

mean = stats.ensemble_mean(xt)
title = f"ECM Ensemble μ, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_x_t_μ{image_postfix}"
label = r"$\mu_{x_t}$"
ensemble_average_plot(mean, title, label, plot_name)

# %%

ensemble_time = numpy.linspace(0, n-1, n)
step = int(len(ensemble_time) / 10)
stationary_time = ensemble_time[::step]
ensemble_std = stats.ensemble_std(xt)
stationary_std = numpy.sqrt([var_xt(arima_params['φ'][0], int(t)) for t in stationary_time])

# %%

title = f"ECM Ensemble σ, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_x_t_σ{image_postfix}"
label = r"$\sigma_{x_t}$"
ensemble_comparison_plot(ensemble_std, ensemble_time, stationary_std, stationary_time, title, label, plot_name)

# %%

ensemble_std = stats.ensemble_std(yt)
stationary_std = numpy.sqrt([var_xt(arima_params['φ'][0], int(t))*ecm_params['β']**2 for t in stationary_time])

# %%

title = f"ECM Ensemble σ, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_y_t_σ{image_postfix}"
label = r"$\sigma_{y_t}$"
ensemble_comparison_plot(ensemble_std, ensemble_time, stationary_std, stationary_time, title, label, plot_name)

# %%

ensemble_cov = stats.ensemble_covariance(xt, yt)
stationary_cov = [var_xt(arima_params['φ'][0], int(t))*ecm_params['β'] for t in stationary_time]

# %%

title = f"ECM Ensemble Covariance, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}, size={m}"
plot_name = f"ecm_properties_ensemble_cov_xt_yt{image_postfix}"
label = r"$Cov(x_t, y_t)$"
ensemble_comparison_plot(ensemble_cov, ensemble_time, stationary_cov, stationary_time, title, label, plot_name)
