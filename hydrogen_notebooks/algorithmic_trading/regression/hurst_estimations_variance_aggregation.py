# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm
from lib import regression as reg
import statsmodels.api as sm

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def aggregated_process(samples, m):
    n = len(samples)
    d = int(n / m)
    agg = numpy.zeros(d)
    for k in range(d):
        for i in range(m):
            j = k*m+i
            agg[k] += samples[j]
        agg[k] = agg[k] / m
    return agg

def aggregated_variance(samples, npts, m_min):
    n = len(samples)
    m_vals = numpy.logspace(numpy.log10(m_min), numpy.log10(n/10.0), npts)
    agg_var = numpy.zeros(npts)
    for i in range(npts):
        m = int(m_vals[i])
        agg = aggregated_process(samples, m)
        agg_mean = numpy.mean(agg)
        d = len(agg)
        for k in range(d):
            agg_var[i] += (agg[k] - agg_mean)**2/(d - 1)
    return agg_var

def m_logspace(nagg, m_min, npts):
    return numpy.logspace(numpy.log10(m_min), numpy.log10(npts/10.0), nagg)

def agg_var_H_estimate(agg_var, m_vals):
    x = numpy.log10(m_vals)
    y = numpy.log10(agg_var)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    results.summary()
    return results.params, results.bse, results.rsquared

def agg_var_plot(agg_var, m_vals, title, plot_name):
    β, σ, r2 = agg_var_H_estimate(agg_var, m_vals)
    h = float(1.0 + β[1]/2.0)
    σ = σ[1] / 2.0
    y_fit = 10**β[0]*m_vals**(β[1])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$Var(X^{m})$")
    axis.set_xlabel(r"$m$")
    axis.set_title(title)
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    x_text = int(0.8*len(m_vals))
    y_text = int(0.4*len(agg_var))
    axis.text(m_vals[x_text], agg_var[y_text],
              r"$\hat{Η}=$" + f"{format(h, '2.3f')}\n" +
              r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.3f')}\n" +
              r"$R^2=$" + f"{format(r2, '2.3f')}",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.loglog(m_vals, agg_var, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=10, label="Simulation")
    axis.loglog(m_vals, y_fit, zorder=5, label=r"$Var(X^{m})=C*m^{2H-2}$")
    axis.legend(bbox_to_anchor=[0.4, 0.4])
    config.save_post_asset(figure, "regression", plot_name)

def agg_process_comparission_multiplot(series, m, ylim, title, plot_name):
    nplot = len(series)
    n = len(series[0])
    figure, axis = pyplot.subplots(nplot, sharex=True, figsize=(12, 9))
    axis[0].set_title(title)
    axis[nplot-1].set_xlabel(r"$t$")
    for i in range(nplot):
        nsample = len(series[i])
        d = int(n/m[i])
        time = numpy.linspace(0, n-1, d)
        axis[i].set_ylabel(r"$x_t$")
        axis[i].set_xlim([0.0, n])
        axis[i].set_ylim(ylim)
        bbox = dict(boxstyle='square',facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
        axis[i].text(time[int(0.9*nsample)], 0.65*ylim[-1], f"m={m[i]}", fontsize=16, bbox=bbox)
        axis[i].plot(time, series[i], lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**12
samples = bm.fbn_fft(H, Δt, npts)

# %%

m = [1, 10, 50]
series = [samples]

title = f"Aggregated Fractional Brownian Noise: Δt={Δt}, H={H}"
plot_name =f"agg_fbn_fft_H_{H}"
series.append(aggregated_process(samples, m[1]))
series.append(aggregated_process(samples, m[2]))
agg_process_comparission_multiplot(series, m, [-3.25, 3.25], title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**12
samples = bm.fbn_fft(H, Δt, npts)

# %%

m = [1, 10, 50]
series = [samples]

title = f"Aggregated Fractional Brownian Noise: Δt={Δt}, H={H}"
plot_name =f"agg_fbn_fft_H_{H}"
series.append(aggregated_process(samples, m[1]))
series.append(aggregated_process(samples, m[2]))
agg_process_comparission_multiplot(series, m, [-3.25, 3.25], title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**16
time = numpy.linspace(0.0, float(npts*Δt), npts)
samples = bm.fbn_fft(H, Δt, npts)

nagg = 100
title = f"Aggregated Variance: H={H}"
plot_name = f"aggregated_variance_fbn_H_{H}_{npts}"
agg_var = aggregated_variance(samples, nagg, 10.0)
m_vals = m_logspace(nagg, 10.0, npts)

# %%

agg_var_plot(agg_var, m_vals, title, plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**16
time = numpy.linspace(0.0, float(npts*Δt), npts)
samples = bm.fbn_fft(H, Δt, npts)

nagg = 100
title = f"Aggregated Variance: H={H}"
plot_name = f"aggregated_variance_fbn_H_{H}_{npts}"
agg_var = aggregated_variance(samples, nagg, 10.0)
m_vals = m_logspace(nagg, 10.0, npts)

# %%

agg_var_plot(agg_var, m_vals, title, plot_name)
