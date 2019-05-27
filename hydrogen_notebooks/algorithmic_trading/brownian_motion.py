# %%
%load_ext autoreload
%autoreload 2

import os
import sys
from lib import brownian_motion
import numpy

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')

# %%

Δt = 0.01
npts = 10000
nsim = 8

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.brownian_motion(Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.brownian_motion(Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion: Δt={Δt}"
brownian_motion.multiplot(samples, time, [5.0, 12.0], title, "brownian_motion_1")

# %%

Δt = 0.01
npts = 10000
nsim = 1000

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.brownian_motion(Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.brownian_motion(Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion: Δt={Δt}"
brownian_motion.multiplot(samples, time, [5.0, 25.0], title, "brownian_motion_2")

# %%

Δt = 0.01
npts = 10000
samples = brownian_motion.brownian_motion(Δt, npts)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion: Δt={Δt}"
brownian_motion.plot(samples, time, title, "brownian_motion_3")

# %%

max_lag = 5000
title = f"Brownian Motion Autocorrelation: Δt={Δt}, npts={npts}"
brownian_motion.autocor(title, samples, Δt, max_lag, "brownian_motion_autocorrelation_1")

# %%

Δt = 0.01
npts = 10000
nsim = 8
μ = 0.1
σ = 0.1

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.brownian_motion_with_drift(μ, σ, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.brownian_motion_with_drift(μ, σ, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion with Drift: Δt={Δt}, μ={μ}, σ={σ}"
brownian_motion.multiplot(samples, time, [5.0, 8.0], title, "brownian_motion_with_drift_1")

# %%

Δt = 0.01
npts = 10000
nsim = 1000
μ = 0.1
σ = 0.1

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.brownian_motion_with_drift(μ, σ, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.brownian_motion_with_drift(μ, σ, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Brownian Motion with Drift: Δt={Δt}, μ={μ}, σ={σ}"
brownian_motion.multiplot(samples, time, [5.0, 8.0], title, "brownian_motion_with_drift_2")

# %%

Δt = 0.01
npts = 10000
nsim = 8
μ = 0.025
σ = 0.15
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion: Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
brownian_motion.multiplot(samples, time, [5.0, 60.0], title, "geometric_brownian_motion_1")


# %%

Δt = 0.01
npts = 10000
nsim = 1000
μ = 0.025
σ = 0.15
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion: Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
brownian_motion.multiplot(samples, time, [5.0, 1000.0], title, "geometric_brownian_motion_2")

# %%

Δt = 0.01
npts = 100
nsim = 5000
μ = 0.0
σ = 1.0
s0 = 1.0

for i in range(nsim):
    if i == 0:
        samples = numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)])
    else:
        samples = numpy.append(samples, numpy.array([brownian_motion.geometric_brownian_motion(μ, σ, s0, Δt, npts)]), axis=0)
time = numpy.linspace(0.0, float(npts-1)*Δt, npts)
title = f"Geometric Brownian Motion: Δt={Δt}, μ={μ}, σ={σ}, "+r"$S_0$" + f"={s0}"
brownian_motion.multiplot(samples, time, [0.2, 30.0], title, "geometric_brownian_motion_3")
