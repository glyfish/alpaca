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

Δt = 1.0
npts = 1000
b = int(numpy.ceil(npts**(1.5)))
time = numpy.linspace(0.0, float(npts)*Δt, npts + 1)
B1 = brownian_motion.brownian_noise(b)
B2 = brownian_motion.brownian_noise(npts+1)

# %%

samples = brownian_motion.brownian_motion_from_noise(B2)
brownian_motion.plot(samples, time, "Brownian Motion From Noise", "brownian_motion_from_noise")

# %%

H = 0.5
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts, B1=B1, B2=B2)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_comparision_H_0.5")

# %%

H = 0.9
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts, B1=B1, B2=B2)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_comparision_H_0.9")

# %%

H = 0.2
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts, B1=B1, B2=B2)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_comparision_H_0.2")

# %%

H = 0.95
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.95")

# %%

H = 0.9
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.9")

# %%

H = 0.8
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.8")

# %%

H = 0.6
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.6")

# %%

H = 0.5
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.5")

# %%

H = 0.4
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.4")

# %%

H = 0.2
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.2")

# %%

H = 0.1
samples = brownian_motion.fb_motion_riemann_sum(H, Δt, npts)
time = numpy.linspace(0.0, float(npts)*Δt, npts+1)
title = f"Fractional Brownian Motion: Δt={Δt}, H={H}"
brownian_motion.plot(samples, time, title, "fbm_riemann_sum_H_0.1")

# %%

H_vals = [0.2, 0.4, 0.45, 0.5, 0.7, 0.8, 0.9, 0.95]
samples = numpy.array([brownian_motion.fb_motion_riemann_sum(H_vals[0], Δt, npts)])
for H in H_vals[1:]:
    samples = numpy.append(samples, numpy.array([brownian_motion.fb_motion_riemann_sum(H, Δt, npts)]), axis=0)

# %%

labels = [f"H={format(H, '1.2f')}" for H in H_vals]
title = f"Fractional Brownian Motion Comparison"
brownian_motion.comparison_multiplot(samples, time, labels, (0.1, 0.75), title, "fbm_riemann_sum_H_comparison")
