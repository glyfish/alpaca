# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import stats
from lib import brownian_motion as bm

wd = os.getcwd()

yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def ensemble_plot(samples, time, text_pos, title, plot_name):
    nsim, npts = samples.shape
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples[:,-1]), '2.2f')}\nσ={format(numpy.std(samples[:,-1]), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    for i in range(nsim):
        axis.plot(time, samples[i], lw=1)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def ensemble_mean_plot(mean, time, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, mean)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def ensemble_std_plot(H, std, time, lengend_location, title, plot_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    step = int(len(time) / 10)
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.plot(time, std, label="Ensemble Average")
    axis.plot(time[::step], time[::step]**H, label=r"$t^{H}$", marker='o', linestyle="None", markeredgewidth=1.0, markersize=15.0)
    axis.legend(bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "brownian_motion", plot_name)

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
    config.save_post_asset(figure, "brownian_motion", plot_name)

# %%

Δt = 1.0
npts = 1024
nsims = 500
time = numpy.linspace(0.0, float(npts)*Δt, npts)

# %%

H = 0.5
samples = numpy.array([bm.fbm_fft(H, Δt, npts)])
for i in range(1, nsims):
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

title = f"Fractional Brownian Motion Ensemble, Ensemble Size={nsims}, H = {format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_H_{format(H, '1.1f')}"
ensemble_plot(samples, time, [10.0, 60.0], title, plot_name)

# %%

mean = stats.ensemble_mean(samples)
title = f"Fractional Brownian Motion Ensemble μ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_μ_H_{format(H, '1.1f')}"
ensemble_mean_plot(mean, time, title, plot_name)

# %%

std = stats.ensemble_std(samples)
title = f"Fractional Brownian Motion Ensemble σ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_σ_H_{format(H, '1.1f')}"
ensemble_std_plot(H, std, time, [0.35, 0.85], title, plot_name)

# %%

ac = stats.ensemble_autocorrelation(bm.to_noise(samples))
title = f"Fractional Brownian Noise Ensemble γ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_noise_ensemble_γ_H_{format(H, '1.1f')}"
ensemble_autocorrelation_plot(H, ac[:200], time[:200], [0.45, 0.3], title, plot_name)

# %%

H = 0.8
samples = numpy.array([bm.fbm_fft(H, Δt, npts)])
for i in range(1, nsims):
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

title = f"Fractional Brownian Motion Ensemble, Ensemble Size={nsims}, H = {format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_H_{format(H, '1.1f')}"
ensemble_plot(samples, time, [100.0, 500.0], title, plot_name)

# %%

mean = stats.ensemble_mean(samples)
title = f"Fractional Brownian Motion Ensemble μ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_μ_H_{format(H, '1.1f')}"
ensemble_mean_plot(mean, time, title, plot_name)

# %%

std = stats.ensemble_std(samples)
title = f"Fractional Brownian Motion Ensemble σ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_σ_H_{format(H, '1.1f')}"
ensemble_std_plot(H, std, time, [0.35, 0.85], title, plot_name)

# %%

ac = stats.ensemble_autocorrelation(bm.to_noise(samples))
title = f"Fractional Brownian Noise Ensemble γ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_noise_ensemble_γ_H_{format(H, '1.1f')}"
ensemble_autocorrelation_plot(H, ac[:200], time[:200], [0.45, 0.85], title, plot_name)

# %%

H = 0.7
samples = numpy.array([bm.fbm_fft(H, Δt, npts)])
for i in range(1, nsims):
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

title = f"Fractional Brownian Motion Ensemble, Ensemble Size={nsims}, H = {format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_H_{format(H, '1.1f')}"
ensemble_plot(samples, time, [100.0, 250.0], title, plot_name)

# %%

mean = stats.ensemble_mean(samples)
title = f"Fractional Brownian Motion Ensemble μ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_μ_H_{format(H, '1.1f')}"
ensemble_mean_plot(mean, time, title, plot_name)

# %%

std = stats.ensemble_std(samples)
title = f"Fractional Brownian Motion Ensemble σ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_σ_H_{format(H, '1.1f')}"
ensemble_std_plot(H, std, time, [0.35, 0.85], title, plot_name)

# %%

ac = stats.ensemble_autocorrelation(bm.to_noise(samples))
title = f"Fractional Brownian Noise Ensemble γ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_noise_ensemble_γ_H_{format(H, '1.1f')}"
ensemble_autocorrelation_plot(H, ac[:200], time[:200], [0.65, 0.85], title, plot_name)

# %%

H = 0.3
samples = numpy.array([bm.fbm_fft(H, Δt, npts)])
for i in range(1, nsims):
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

title = f"Fractional Brownian Motion Ensemble, Ensemble Size={nsims}, H = {format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_H_{format(H, '1.1f')}"
ensemble_plot(samples, time, [50.0, -30.0], title, plot_name)

# %%

mean = stats.ensemble_mean(samples)
title = f"Fractional Brownian Motion Ensemble μ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_μ_H_{format(H, '1.1f')}"
ensemble_mean_plot(mean, time, title, plot_name)

# %%

std = stats.ensemble_std(samples)
title = f"Fractional Brownian Motion Ensemble σ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_σ_H_{format(H, '1.1f')}"
ensemble_std_plot(H, std, time, [0.35, 0.9], title, plot_name)

# %%

ac = stats.ensemble_autocorrelation(bm.to_noise(samples))
title = f"Fractional Brownian Noise Ensemble γ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_noise_ensemble_γ_H_{format(H, '1.1f')}"
ensemble_autocorrelation_plot(H, ac[:200], time[:200], [0.5, 0.85], title, plot_name)

# %%

H = 0.9
npts = 4098
time = numpy.linspace(0.0, float(npts)*Δt, npts)

samples = numpy.array([bm.fbm_fft(H, Δt, npts)])
for i in range(1, nsims):
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

title = f"Fractional Brownian Motion Ensemble, Ensemble Size={nsims}, H = {format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_H_{format(H, '1.1f')}"
ensemble_plot(samples, time, [100.0, 3000.0], title, plot_name)

# %%

mean = stats.ensemble_mean(samples)
title = f"Fractional Brownian Motion Ensemble μ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_μ_H_{format(H, '1.1f')}"
ensemble_mean_plot(mean, time, title, plot_name)

# %%

std = stats.ensemble_std(samples)
title = f"Fractional Brownian Motion Ensemble σ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_motion_ensemble_σ_H_{format(H, '1.1f')}"
ensemble_std_plot(H, std, time, [0.35, 0.85], title, plot_name)

# %%

ac = stats.ensemble_autocorrelation(bm.to_noise(samples))
title = f"Fractional Brownian Noise Ensemble γ, Ensemble Size={nsims}, H={format(H, '1.1f')}"
plot_name = f"fractional_brownian_noise_ensemble_γ_H_{format(H, '1.1f')}"
ensemble_autocorrelation_plot(H, ac[:200], time[:200], [0.45, 0.4], title, plot_name)
