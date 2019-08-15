# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

def fbm_covariance_plot(H_vals, s, time, file_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(r"Fraction Brownian Motion Covariance, $\frac{1}{2}[t^{2H} + s^{2H} - \mid t-s \mid^{2H}]$" + f", s={format(s, '1.2f')}")

    for H in H_vals:
        axis.plot(time, brownian_motion.fbm_covariance(H, s, time), label=f"H={format(H, '1.2f')}")

    axis.legend(ncol=2)
    config.save_post_asset(figure, "brownian_motion", file_name)

def fbm_autocovariance_plot(H_vals, time, lengend_location, file_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(r"Fraction Brownian Motion Autocovariance, $\frac{1}{2}[(n-1)^{2H} + (n+1)^{2H} - 2n^{2H}]$")

    for H in H_vals:
        axis.plot(time, brownian_motion.fbm_autocovariance(H, time), label=f"H={format(H, '1.2f')}")

    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
    config.save_post_asset(figure, "brownian_motion", file_name)

def fbm_autocvariance_limit(H_vals, time, lengend_location, file_name):
    figure, axis = pyplot.subplots(figsize=(12, 8))
    axis.set_xlabel("Time")
    axis.set_ylabel("Value")
    axis.set_title(r"Fraction Brownian Motion Autocovariance for $n\gg1$, $H(2H-1)n^{2H-2}$")

    for H in H_vals:
        axis.plot(time, brownian_motion.fbm_autocovariance_limit(H, time), label=f"H={format(H, '1.2f')}")

    axis.legend(ncol=2, bbox_to_anchor=lengend_location)
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
    axis.plot(time, brownian_motion.fbm_variance(H, time), label=f"H={format(H, '1.2f')}")

axis.legend(ncol=2, bbox_to_anchor=(0.6, 0.35))
config.save_post_asset(figure, "brownian_motion", "fractional_brownian_motion_variance")

# %%

s = 0.25
fbm_covariance_plot(H_vals, s, time, "fractional_brownian_motion_covariance_0.25")

# %%

s = 0.5
fbm_covariance_plot(H_vals, s, time, "fractional_brownian_motion_covariance_0.5")

# %%

s = 1.0
fbm_covariance_plot(H_vals, s, time, "fractional_brownian_motion_covariance_1.0")

# %%

s = 1.0
tmax = 1000.0
Δt = 0.01
nsteps = int(tmax/Δt)
time = numpy.linspace(0.0, tmax, nsteps)
H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

fbm_covariance_plot(H_vals, s, time, "fractional_brownian_motion_covariance_long_time_low_H_1.0")

# %%

s = 1.0
tmax = 1000.0
Δt = 0.01
nsteps = int(tmax/Δt)
time = numpy.linspace(0.0, tmax, nsteps)
H_vals = [0.6, 0.7, 0.8, 0.9, 1.0]

fbm_covariance_plot(H_vals, s, time, "fractional_brownian_motion_covariance_long_time_high_H_1.0")

# %%

tmax = 10.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocovariance_plot(H_vals, time, (0.8, 0.75), "fractional_brownian_motion_autocovariance_H_lt_eq_0.5")


# %%

tmax = 10.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocovariance_plot(H_vals, time, (0.6125, 0.75), "fractional_brownian_motion_autocovariance_H_gt_eq_0.5")

# %%

tmax = 1000.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocovariance_plot(H_vals, time, (0.8, 0.75), "fractional_brownian_motion_autocovariance_H_lt_eq_0.5_long_time")


# %%

tmax = 1000.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocovariance_plot(H_vals, time, (0.6125, 0.75), "fractional_brownian_motion_autocovariance_H_gt_eq_0.5_long_time")

# %%

tmax = 1000.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocvariance_limit(H_vals, time, (0.6, 0.525), "fractional_brownian_motion_autocovariance_limit_H_lt_eq_0.5")

# %%

tmax = 1000.0
Δt = 1.0
nsteps = int(tmax/Δt)

H_vals = [0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
time = numpy.linspace(1.0, tmax, nsteps)

fbm_autocvariance_limit(H_vals, time, (0.6125, 0.75), "fractional_brownian_motion_autocovariance_limit_H_gt_eq_0.5")
