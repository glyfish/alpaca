# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion as bm

wd = os.getcwd()

yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

Δt = 1.0
npts = 1000
time = numpy.linspace(0.0, float(npts)*Δt, npts)
dB = bm.brownian_noise(2*npts)

title = f"Brownian Noise"
bm.plot(dB[:n], time, title, "brwonian_noise_fft")

# %%

H = 0.5
samples = bm.fbn_fft(H, Δt, npts, dB=dB)
title = f"FFT Fractional Brownian Noise: Δt={Δt}, H={H}"
bm.plot(samples, time,title, "fbn_fft_H_0.5")

# %%

H = 0.9
samples = bm.fbn_fft(H, Δt, npts, dB=dB)
title = f"FFT Fractional Brownian Noise: Δt={Δt}, H={H}"
bm.plot(samples, time,title, "fbn_fft_H_0.9")

# %%

H = 0.2
samples = bm.fbn_fft(H, Δt, npts, dB=dB)
title = f"FFT Fractional Brownian Noise: Δt={Δt}, H={H}"
bm.plot(samples, time,title, "fbn_fft_H_0.2")

# %%

H_vals = [0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
samples = numpy.array([bm.fbm_fft(H_vals[0], Δt, npts)])
for H in H_vals[1:]:
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

labels = [f"H={format(H, '1.2f')}" for H in H_vals]
title = f"FFT Fractional Brownian Motion Comparison"
bm.comparison_multiplot(samples, time, labels, (0.375, 0.75), title, "fbm_fft_H_gt_0.5_comparison")

# %%

H_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5]
samples = numpy.array([bm.fbm_fft(H_vals[0], Δt, npts)])
for H in H_vals[1:]:
    samples = numpy.append(samples, numpy.array([bm.fbm_fft(H, Δt, npts)]), axis=0)

# %%

labels = [f"H={format(H, '1.2f')}" for H in H_vals]
title = f"FFT Fractional Brownian Motion Comparison"
bm.comparison_multiplot(samples, time, labels, (0.38, 0.275), title, "fbm_fft_H_leq_0.5_comparison")
