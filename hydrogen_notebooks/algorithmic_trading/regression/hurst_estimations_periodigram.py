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
from lib import stats
import statsmodels.api as sm

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def power_spec_H_estimate(power_spec, freq):
    x = numpy.log10(freq)
    y = numpy.log10(power_spec)
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    results.summary()
    return results.params, results.bse

def plot(sample, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"t")
    axis.set_title(title)
    axis.plot(t, sample)
    config.save_post_asset(figure, "regression", plot_name)

def periodogram_plot(power_spec, freq, title, plot_name):
    β, σ = power_spec_H_estimate(power_spec, freq)
    h = float(1.0 + β[1]/2.0)
    y_fit = 10**β[0]*m_vals**(β[1])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$Var(X^{m})$")
    axis.set_xlabel(r"$m$")
    axis.set_title(title)
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    x_text = int(0.8*len(freq))
    y_text = int(0.4*len(power_spec))
    axis.text(freq[x_text], agg_var[y_text],
              r"$\hat{Η}=$" + f"{format(h, '2.3f')}\n" +
              r"$\sigma_{\hat{H}}=$" + f"{format(σ[1], '2.3f')}",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.loglog(m_vals, power_spec, marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=10, label="Simulation")
    axis.loglog(m_vals, y_fit, zorder=5, label=r"$Var(X^{m})=C*m^{2H-2}$")
    axis.legend(bbox_to_anchor=[0.4, 0.4])
    config.save_post_asset(figure, "regression", plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**16
samples = bm.fbn_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)

# %%

title = f"Aggregated Fractional Brownian Noise: Δt={Δt}, H={H}"
plot_name =f"peridigram_fbn_fft_H_{H}"
plot(samples, time, title, plot_name)

# %%

ps = stats.power_spectrum(samples)

len(ps)

numpy.sum(ps)
