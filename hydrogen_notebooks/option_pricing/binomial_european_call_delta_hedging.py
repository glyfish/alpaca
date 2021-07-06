# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from scipy.stats import binom

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
def qrn(U, D, R):
    return (R - D) / (U - D)

def qrn1(q, U, R):
    return q*(1.0 + U) / (1.0 + R)

def binomial_tail_cdf(l, n, p):
    return 1.0 - binom.cdf(l, n, p)

def cutoff(S0, U, D, K, n):
    for i in range(0, n + 1):
        iU = (1.0 + U)**i
        iD = (1.0 + D)**(n - i)
        payoff = S0*iU*iD - K
        if payoff > 0:
            return i
    return n + 1

def european_call_payoff(U, D, R, S0, K, n):
    l = cutoff(S0, U, D, K, n)
    q = qrn(U, D, R)
    q1 = qrn1(q, U, R)
    Ψq = binomial_tail_cdf(l - 1, n, q)
    Ψq1 = binomial_tail_cdf(l - 1, n, q1)
    return S0*Ψq1 - K*(1 + R)**(-n)*Ψq

# %%

n = 3
U = 0.2
D = -0.1
R = 0.1
S0 = 100.0
K = 105.0

# %%

q = qrn(U, D, R)
q1 = qrn1(q, U, R)
l = cutoff(S0, U, D, K, n)
Ψq = binomial_tail_cdf(l - 1, n, q)
Ψq1 = binomial_tail_cdf(l - 1, n, q1)

q, q1, l, Ψq, Ψq1
binom.cdf(l, n, q)

# %
# t = 0

european_call_payoff(U, D, R, S0, K, n)

# %%
# Delta hedge
# t = 1

S1U = S0*(1.0 + U)
S1D = S0*(1.0 + D)

european_call_payoff(U, D, R, S1U, K, n - 1)
european_call_payoff(U, D, R, S1D, K, n - 1)

# t = 2
