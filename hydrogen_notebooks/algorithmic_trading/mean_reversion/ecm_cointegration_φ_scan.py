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

def ols_correlation_estimate(xt, yt):
    xt = sm.add_constant(xt)
    model = sm.OLS(yt, xt)
    results = model.fit()
    print(results.summary())
    return results.params, results.rsquared, results.bse

def ecm_sample_generate(arima_params, ecm_params, n):
    xt = arima.arima_generate_sample(arima_params["φ"], arima_params["δ"], arima_params["d"], n)
    yt = numpy.zeros(n)
    ξt = numpy.random.normal(0.0, 1.0, n)
    δ = ecm_params["δ"]
    λ = ecm_params["λ"]
    α = ecm_params["α"]
    β = ecm_params["β"]
    γ = ecm_params["γ"]
    for i in range(1, n):
        Δxt = xt[i] - xt[i-1]
        Δyt = δ + γ*Δxt + λ*(yt[i-1] - α - β*xt[i-1]) + ξt[i]
        yt[i] = Δyt + yt[i-1]
    return xt, yt

# %%

arima_params = {"φ": numpy.array([0.9]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)
# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.-1]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([-0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.1]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([-0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([-0.9]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([-0.1]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "γ": 0.5, "λ": -0.5, "α": 0.0, "β": 0.5, "σ": 1.0}
n = 1000
image_postfix = f"_φ_{format(arima_params['φ'][0], '1.1f')}_β_{format(ecm_params['β'], '1.1f')}_λ_{format(ecm_params['λ'], '1.1f')}_γ_{format(ecm_params['γ'], '1.1f')}_σ_{format(ecm_params['σ'], '1.1f')}"

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation{image_postfix}"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation{image_postfix}"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual{image_postfix}"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}, " + r"$\gamma=$" + f"{format(ecm_params['γ'], '2.2f')}, " + r"$\sigma=$" + f"{format(ecm_params['σ'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_acf_pacf{image_postfix}"
max_lag = 15
ylim = [-0.1, 1.1]
arima.acf_pcf_plot(title, εt, ylim, max_lag, plot_name)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())
