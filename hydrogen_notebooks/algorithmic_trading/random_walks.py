# %%
%load_ext autoreload
%autoreload 2

import os
import sys
from datetime import datetime
import backtrader
from matplotlib import pyplot
from lib import config
import numpy

pyplot.style.use(config.glyfish_style)
wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')

# %%

def brownian_motion(Δt, n):
    σ = numpy.sqrt(Δt)
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + σ * Δ
    return samples

def brownian_motion_with_drift(μ, σ, Δt, n):
    samples = numpy.zeros(n)
    for i in range(1, n):
        Δ = numpy.random.normal()
        samples[i] = samples[i-1] + (σ * Δ * numpy.sqrt(Δt)) + (μ * Δt)
    return samples

def geometric_brownian_motion(μ, σ, s0, Δt, n):
    samples = brownian_motion_with_drift(μ, σ, Δt, n)
    return s0*numpy.exp(samples)

def multiplot(samples, time, text_pos, title, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(title)
    stats=f"Simulation Stats\n\nμ={format(numpy.mean(samples[:,-1]), '2.2f')}\nσ={format(numpy.std(samples[:,-1]), '2.2f')}"
    bbox = dict(boxstyle='square,pad=1', facecolor="#FEFCEC", edgecolor="#FEFCEC", alpha=0.75)
    axis.text(text_pos[0], text_pos[1], stats, fontsize=15, bbox=bbox)
    for i in range(nplot):
        axis.plot(time, samples[i], lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

def plot(samples, time, text_pos, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "mean_reversion", plot_name)

# %%

Δt = 0.01
npts = 10000
nsim = 8

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion(Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion(Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion; Δt={Δt}"
multiplot(samples, time, [5.0, 12.0], title, "brownian_motion_1")

# %%

Δt = 0.01
npts = 10000
nsim = 1000

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion(Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion(Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion; Δt={Δt}"
multiplot(samples, time, [5.0, 25.0], title, "brownian_motion_2")

# %%

Δt = 0.01
npts = 10000
samples = brownian_motion(Δt, npts)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion; Δt={Δt}"
plot(samples, time, title, "brownian_motion_3")

# %%

Δt = 0.01
npts = 10000
nsim = 8
μ = 0.1
σ = 0.1

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion_with_drift(μ, σ, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion_with_drift(μ, σ, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion with Drift; Δt={Δt}, μ={μ}, σ={σ}"
multiplot(samples, time, [5.0, 8.0], title, "brownian_motion_with_drift_1")

# %%

Δt = 0.01
npts = 10000
nsim = 1000
μ = 0.1
σ = 0.1

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion_with_drift(μ, σ, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion_with_drift(μ, σ, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion with Drift; Δt={Δt}, μ={μ}, σ={σ}"
multiplot(samples, time, [5.0, 8.0], title, "brownian_motion_with_drift_2")

# %%

Δt = 0.01
npts = 10000
nsim = 8
μ = 0.025
σ = 0.15
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion; Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
multiplot(samples, time, [5.0, 60.0], title, "geometric_brownian_motion_1")


# %%

Δt = 0.01
npts = 10000
nsim = 1000
μ = 0.025
σ = 0.15
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion; Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
multiplot(samples, time, [5.0, 1000.0], title, "geometric_brownian_motion_2")

# %%

Δt = 0.01
npts = 10000
nsim = 10000
μ = 0.025
σ = 0.15
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion; Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
multiplot(samples, time, [5.0, 1000.0], title, "geometric_brownian_motion_3")
