# %%
%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def fbm_variance(H, time):
    return time**(2.0*H)

def fbm_autocorrelation(H, s, time):
    return 0.5*(time**(2.0*H) + s**(2.0*H) - numpy.abs(time - s)**(2.0*H))

def fbm_autocorrelation_plot(H, s, time, file_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(r"Fraction Brownian Motion Autocorrelation, $\frac{1}{2}[t^{2H} + s^{2H} - \mid t-s \mid^{2H}]$" + f", s={format(s, '1.2f')}")

    for H in H_vals:
        axis.plot(time, fbm_autocorrelation(H, s, time), label=f"H={format(H, '1.2f')}")

    axis.legend(ncol=2, bbox_to_anchor=(0.6, 0.35))
    config.save_post_asset(figure, "brownian_motion", file_name)

# %%

tmax = 1.0
Δt = 0.0001
nsteps = int(tmax/Δt)

H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
time = numpy.linspace(0.0, tmax, nsteps)

# %%

figure, axis = pyplot.subplots(figsize=(12, 8))
axis.set_xlabel("Time")
axis.set_ylabel("Value")
axis.set_title(r"Fraction Brownian Motion Variance, $t^{2H}$")

for H in H_vals:
    axis.plot(time, fbm_variance(H, time), label=f"H={format(H, '1.2f')}")

axis.legend(ncol=2, bbox_to_anchor=(0.6, 0.35))
config.save_post_asset(figure, "brownian_motion", "fractional_brownian_motion_variance")

# %%

s = 0.5
fbm_autocorrelation_plot(H, s, time, "fractional_brownian_motion_autocorrelation_0.5")

# %%

s = 1.0
fbm_autocorrelation_plot(H, s, time, "fractional_brownian_motion_autocorrelation_1.0")


# %%

s = 1.0
tmax = 1000.0
Δt = 0.01
nsteps = int(tmax/Δt)
time = numpy.linspace(0.0, tmax, nsteps)

# %%

fbm_autocorrelation_plot(H, s, time, "fractional_brownian_motion_autocorrelation_long_time_1.0")
