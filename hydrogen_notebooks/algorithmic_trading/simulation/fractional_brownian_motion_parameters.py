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

def variance(H, tmax, Δt):
    nsteps = int(tmax/Δt)
    time = numpy.linspace(0.0, tmax, nsteps)
    return time**(2.0*H)

def autcorrelation(H, s, tmax, Δt):
    nsteps = int(tmax/Δt)
    time = numpy.linspace(0.0, tmax, steps)
    return 0.5*(time**(2.0*H) + s**(2.0*H) + (time - s)**(2.0*H))

# %%

tmax = 10.0
Δt = 0.1
nsteps = int(tmax/Δt)

H_vals = numpy.linspace(0.05, 1.0, 20)

figure, axis = pyplot.subplots(figsize=(12, 8))
axis.set_xlabel("Time")
axis.set_ylabel("Value")
axis.set_title("Fraction Brownian Motion Variance")

for H in H_vals:
    time = numpy.linspace(0.0, tmax, nsteps)
    variance = variance(H, tmax, Δt)
    axis.plot(time, variance)

config.save_post_asset(figure, "brownian_motion", "fractional_brownian_motion_variance")
