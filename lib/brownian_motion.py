import numpy
from matplotlib import pyplot
from lib import config
from lib import stats

pyplot.style.use(config.glyfish_style)

# Brownian Motion Simulations

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

# Plots

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
    config.save_post_asset(figure, "brownian_motion", plot_name)

def plot(samples, time, title, plot_name):
    nplot = len(samples)
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(title)
    axis.plot(time, samples, lw=1)
    config.save_post_asset(figure, "brownian_motion", plot_name)

def autocor_coef(title, samples, Δt, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, Δt*max_lag])
    axis.set_ylim([-1.05, 1.0])
    ac = stats.autocorrelate(samples)
    axis.plot(Δt*numpy.array(range(max_lag)), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "brownian_motion", plot)

def autocor(title, samples, Δt, max_lag, plot):
    figure, axis = pyplot.subplots(figsize=(10, 7))
    axis.set_title(title)
    axis.set_ylabel(r"$\gamma_{\tau}$")
    axis.set_xlabel("Time Lag (τ)")
    axis.set_xlim([0, Δt*max_lag])
    axis.set_ylim([-1.05, 1.0])
    ac = stats.autocorrelate(samples)
    axis.plot(Δt*numpy.array(range(max_lag)), numpy.real(ac[:max_lag]))
    config.save_post_asset(figure, "brownian_motion", plot)
