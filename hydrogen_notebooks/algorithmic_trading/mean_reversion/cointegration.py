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
    δ = ecm_params["δ"]
    λ = ecm_params["λ"]
    α = ecm_params["α"]
    β = ecm_params["β"]
    for i in range(1, n):
        Δxt = xt[i] - xt[i-1]
        Δyt = δ + Δxt - λ*(yt[i-1] - α - β*xt[i-1])
        yt[i] = Δyt + yt[i-1]
    return xt, yt

# %%

φ = numpy.array([0.8])
δ = numpy.array([])
d = 1
n = 1000

xt = arima.arima_generate_sample(φ, δ, d, n)
yt = arima.arima_generate_sample(φ, δ, d, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"Spurious Correlation of Independent I(1) Time timeseries_plot"
plot_name = f"cointegration_I(1)_spurious_correlation"
corrletation_plot(xt, yt, params, err, rsquard, [0.7, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = r"Series Comparison for $I(1)$, " + r"$\phi=$"+f"{numpy.array2string(φ, precision=2, separator=',')}, " + r"$\delta=$"+f"{numpy.array2string(δ, precision=2, separator=',')}"
plot_name = "cointegration_I(1)_sum_comparison_1"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

arima_params = {"φ": numpy.array([0.8]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 1.0, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.8_β_0.5_λ_1.0"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.8_β_0.5_λ_1.0"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.8_β_0.5_λ_1.0"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.8]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 0.5, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.8_β_0.5_λ_0.5"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.8_β_0.5_λ_0.5"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.8_β_0.5_λ_0.5"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.8]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 0.1, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.8_β_0.5_λ_0.1"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.8_β_0.5_λ_0.1"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.8_β_0.5_λ_0.1"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.8]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 0.5, "α": 0.0, "β": 0.25}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.8_β_0.25_λ_0.5"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.8_β_0.25_λ_0.5"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.8_β_0.25_λ_0.5"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 1.0, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.5_β_0.5_λ_1.0"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.5_β_0.25_λ_0.5"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.5_β_0.25_λ_0.5"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 0.5, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.5_β_0.5_λ_0.5"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.5_β_0.5_λ_0.5"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.5_β_0.5_λ_0.5"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())

# %%

arima_params = {"φ": numpy.array([0.5]), "δ": numpy.array([]), "d": 1}
ecm_params = {"δ": 0.0, "λ": 0.25, "α": 0.0, "β": 0.5}
n = 1000

xt, yt = ecm_sample_generate(arima_params, ecm_params, n)

# %%

params, rsquard, err = ols_correlation_estimate(xt, yt)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_corrleation_φ_0.5_β_0.5_λ_0.25"
corrletation_plot(xt, yt, params, err, rsquard, [0.85, 0.5], title, plot_name)

# %%

εt = yt - params[0] - params[1]*xt

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_φ_0.5_β_0.5_λ_0.25"
labels = [r"$x_t$", r"$y_t$", r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([xt, yt, εt])

comparison_plot(title, samples, labels, plot_name)

# %%

title = f"ECM Simulation, " + r"$\phi=$" + f"{numpy.array2string(arima_params['φ'], precision=2, separator=',')}, " + r"$\lambda=$" + f"{format(ecm_params['λ'], '2.2f')}, " + r"$\beta=$" + f"{format(ecm_params['β'], '2.2f')}"
plot_name = f"cointegration_ecm_simulation_residual_φ_0.5_β_0.5_λ_0.25"
labels = [r"$\varepsilon_t = y_{t}-\hat{\alpha}-\hat{\beta}x_{t}$"]
samples = numpy.array([εt])

comparison_plot(title, samples, labels, plot_name)

# %%

arima.adf_report(εt)

# %%

model_fit = arima.arma_estimate_parameters(εt, (1, 0))
print(model_fit.summary())
