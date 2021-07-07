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

def delta(CU, CD, SU, SD):
    return (CU - CD) / (SU - SD)

def init_borrow(S0, C0, x):
    return C0 - S0 * x

def borrow(y, R, x1, x2, S):
    return y * (1 + R) + (x1 - x2) * S

def portfolio_value(x, S, y):
    return x * S + y

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

C0 = european_call_payoff(U, D, R, S0, K, n)

# %%
# Delta hedge
# t = 0

S1U = S0*(1.0 + U)
S1D = S0*(1.0 + D)

C1U = european_call_payoff(U, D, R, S1U, K, n - 1)
C1D = european_call_payoff(U, D, R, S1D, K, n - 1)

x1 = delta(C1U, C1D, S1U, S1D)
y1 = init_borrow(S0, C0, x1)

portfolio_value(x1, S0, y1)

# t = 1
# The price goes up S1 = S0*(1+U)

S1 = S0 * (1 + U)

S2U = S1*(1.0 + U)
S2D = S1*(1.0 + D)

C2U = european_call_payoff(U, D, R, S2U, K, n - 2)
C2D = european_call_payoff(U, D, R, S2D, K, n - 2)

x2 = delta(C2U, C2D, S2U, S2D)
y2 = borrow(y1, R, x1, x2, S1)

portfolio_value(x2, S1, y2)

# t = 2
# The price goes down S1 = S0*(1+U)*(1+D)

S2 = S0 * (1 + U) * (1 + D)

S3U = S2*(1.0 + U)
S3D = S2*(1.0 + D)

C3U = european_call_payoff(U, D, R, S3U, K, n - 3)
C3D = european_call_payoff(U, D, R, S3D, K, n - 3)

x3 = delta(C3U, C3D, S3U, S3D)
y3 = borrow(y2, R, x2, x3, S2)

portfolio_value(x3, S2, y3)
