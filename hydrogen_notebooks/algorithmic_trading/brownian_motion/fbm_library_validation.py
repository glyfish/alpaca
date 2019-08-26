# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib.fbm import FBM
from lib import config
from lib import brownian_motion as bm
from lib import stats

wd = os.getcwd()

yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

f = FBM(n=1024, hurst=0.85, length=1024, method='daviesharte')
fbm = f.fbm()
fbn = f.fgn()
times = f.times()

# %%

bm.plot(fbn, times[:1024], "FBM Library FBN H=0.85", "fbm_library_fbn_H_0.85")

# %%

bm.plot(fbm, times, "FBM Library FBM H=0.85", "fbm_library_fbm_H_0.85")

# %%

ac = stats.autocorrelate(fbn).real
bm.plot(ac[:200], times[:200], "FBM Library FBN γ H=0.85", "fbm_library_γ_H_0.85")

# %%

f = FBM(n=1024, hurst=0.9, length=1024, method='daviesharte')
row_component = [f._autocovariance(i) for i in range(1, f.n)]
reverse_component = list(reversed(row_component))
row = [f._autocovariance(0)] + row_component + [0] + reverse_component
row[1025]

# %%

bm_row_component = [bm.fbn_autocorrelation(0.9, i) for i in range(1, f.n)]
bm_reverse_component = list(reversed(bm_row_component))
bm_row = [bm.fbn_autocorrelation(0.9, 0)] + bm_row_component + [0] + bm_reverse_component
bm_row[1025]

# %%

n = 1024
H = 0.9
C = numpy.zeros(2*n)
for i in range(2*n):
    if i == 0:
        C[i] = 1.0
    if i == n:
        C[i] = 0.0
    elif i < n:
        C[i] = bm.fbn_autocorrelation(H, i)
    else:
        C[i] = bm.fbn_autocorrelation(H, 2*n-i)

C[1025]
