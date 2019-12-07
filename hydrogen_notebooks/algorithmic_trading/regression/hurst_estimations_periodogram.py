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
    return results.params, results.bse, results.rsquared

def plot(sample, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$x_t$")
    axis.set_xlabel(r"t")
    axis.set_title(title)
    axis.plot(t, sample, lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

def loglogplot(sample, t, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$F(\omega)$")
    axis.set_xlabel(r"$\omega$")
    axis.set_title(title)
    axis.loglog(t, sample, lw=1.0)
    config.save_post_asset(figure, "regression", plot_name)

def periodogram_plot(power_spec, freq, freq_min, title, plot_name):
    β, σ, r2 = power_spec_H_estimate(power_spec[freq_min:], freq[freq_min:])
    h = float((1.0-β[1])/2.0)
    σ = σ[1] / 2.0
    y_fit = 10**β[0]*freq**(β[1])
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_ylabel(r"$F(\omega)$")
    axis.set_xlabel(r"$\omega$")
    axis.set_title(title)
    bbox = dict(boxstyle='square,pad=1', facecolor="#f7f6e8", edgecolor="#f7f6e8")
    x_text = int(0.2*len(freq))
    axis.text(freq[x_text], power_spec[11],
              r"$\hat{Η}=$" + f"{format(h, '2.3f')}\n" +
              r"$\sigma_{\hat{H}}=$" + f"{format(σ, '2.3f')}\n" +
              r"$R^2=$" + f"{format(r2, '2.3f')}",
              bbox=bbox, fontsize=14.0, zorder=7)
    axis.loglog(freq[::5], power_spec[::5], marker='o', markersize=5.0, linestyle="None", markeredgewidth=1.0, alpha=0.75, zorder=5, label="Simulation")
    axis.loglog(freq, y_fit, zorder=10, label=r"$F(\omega)=C*\omega^{1-2H}$")
    axis.legend(bbox_to_anchor=[0.4, 0.3])
    config.save_post_asset(figure, "regression", plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**10
samples = bm.fbn_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
ps = stats.power_spectrum(samples)
ω = numpy.linspace(1.0, len(ps)+1, len(ps))

# %%

title = f"Fractional Brownian Noise: Δt={Δt}, H={H}"
plot_name =f"periodogram_fbn_fft_H_{H}"
plot(samples, time, title, plot_name)

# %%

title = f"Fractional Brownian Noise Power Spectrum: Δt={Δt}, H={H}"
plot_name =f"periodogram_fbn_fft_power_spectrum_H_{H}"
loglogplot(ps, ω, title, plot_name)

# %%

H = 0.8
Δt = 1.0
npts = 2**16
samples = bm.fbn_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
ps = stats.power_spectrum(samples)
ω = numpy.linspace(1.0, len(ps)+1, len(ps))

# %%

title = f"Fractional Brownian Noise Power Spectrum: Δt={Δt}, H={H}, N={npts}"
plot_name =f"periodogram_fbn_fft_power_spectrum_fit_H_{H}"
periodogram_plot(ps, ω, 20, title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**10
samples = bm.fbn_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
ps = stats.power_spectrum(samples)
ω = numpy.linspace(1.0, len(ps)+1, len(ps))

# %%

title = f"Fractional Brownian Noise: Δt={Δt}, H={H}"
plot_name =f"periodogram_fbn_fft_H_{H}"
plot(samples, time, title, plot_name)

# %%

title = f"Fractional Brownian Noise Power Spectrum: Δt={Δt}, H={H}"
plot_name =f"periodogram_fbn_fft_power_spectrum_H_{H}"
loglogplot(ps, ω, title, plot_name)

# %%

H = 0.4
Δt = 1.0
npts = 2**16
samples = bm.fbn_fft(H, Δt, npts)
time = numpy.linspace(0.0, Δt*npts - 1, npts)
ps = stats.power_spectrum(samples)
ω = numpy.linspace(1.0, len(ps)+1, len(ps))

# %%

title = f"Fractional Brownian Noise Power Spectrum: Δt={Δt}, H={H}, N={npts}"
plot_name =f"periodogram_fbn_fft_power_spectrum_fit_H_{H}"
periodogram_plot(ps, ω, 20, title, plot_name)
