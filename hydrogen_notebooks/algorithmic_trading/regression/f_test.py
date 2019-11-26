# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
from lib import brownian_motion
from lib import regression as reg
from scipy.stats import chi2
from scipy.stats import f
from scipy import special
from statsmodels.tsa import stattools

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%
# Example of single variance hypothesis test

npts = 1001
xmax = 300.0
x = numpy.linspace(0.0001, xmax, npts)
k = 145

# %%

title = r"$\chi^2$"+f" PDF from scipy, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
plot = f"chi_squared_test_chi_squared_{k}_pdf"
reg.distribution_plot(reg.chi_squared_pdf(k)(x), x, title, ylabel, xlabel, plot)

# %%

title = r"$\chi^2$"+f" CDF, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
plot = f"chi_squared_test_chi_squared_{k}_cdf"
reg.distribution_plot(reg.chi_squared_cdf(k)(x), x, title, ylabel, xlabel, plot)

# %%

sx2 = 308.56
σ2 = 225.0
t = k*sx2/σ2
t
reg.chi_squared_cdf(k)(113.65)
reg.chi_squared_cdf(k)(180.23)
reg.chi_squared_cdf(k)(174.3)
1.0-reg.chi_squared_cdf(k)(t)
1.0-chi2.cdf(t, k)

# %%

npts = 1001
xmax = 10.0
x = numpy.linspace(0.0001, xmax, npts)

# %%

a = 10
b = 10
title = f"F-Distibution PDF from scipy, Number of Degrees of Freedom: {a}, {b}"
ylabel = r"$f(t;10,10)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_{a},{b}_pdf_scipi"
reg.distribution_plot(f.pdf(x, a, b), x, title, ylabel, xlabel, plot)

# %%

title = f"F-Distibution CDF from scipy, Number of Degrees of Freedom: {a}, {b}"
ylabel = r"$F(t;10,10)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_{a},{b}_cdf_scipi"
reg.distribution_plot(f.cdf(x, a, b), x, title, ylabel, xlabel, plot)

# %%

a = 10
b = 10
title = f"F-Distibution PDF, Number of Degrees of Freedom: {a}, {b}"
ylabel = r"$f(t;10,10)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_{a},{b}_cdf"
reg.distribution_plot(reg.f_pdf(a, b)(x), x, title, ylabel, xlabel, plot)

# %%

title = f"F-Distibution CDF, Number of Degrees of Freedom: {a}, {b}"
ylabel = r"$F(t;10,10)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_{a},{b}_cdf"
reg.distribution_plot(reg.f_cdf(a, b)(x), x, title, ylabel, xlabel, plot)

# %%

a_vals = [1, 5, 10, 50, 100]
b = 10

fx = [reg.f_pdf(a,b)(x) for a in a_vals]
labels = [f"(a,b)=({a},{b})" for a in a_vals]
title = f"F-Distibution PDF for Range of Degrees of Freedom"
ylabel = r"$f(t;a,b)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_pdf_a_scan_{b}"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.4, 0.7], [0.0, 1.1], title, plot)

# %%

a = 10
b_vals = [1, 5, 10, 50, 100]

fx = [reg.f_pdf(a,b)(x) for b in b_vals]
labels = [f"(a,b)=({a},{b})" for b in b_vals]
title = f"F-Distibution PDF for Range of Degrees of Freedom"
ylabel = r"f(t;a,b)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_pdf_b_scan_{a}"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.4, 0.7], [0.0, 1.1], title, plot)

# %%

a_vals = [1, 5, 10, 50, 100]
b = 10

fx = [reg.f_cdf(a,b)(x) for a in a_vals]
labels = [f"(a,b)=({a},{b})" for a in a_vals]
title = f"F-Distibution PDF for Range of Degrees of Freedom"
ylabel = r"$F(t;a,b)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_cdf_a_scan_{b}"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.4, 0.7], [0.0, 1.1], title, plot)

# %%

a = 10
b_vals = [1, 5, 10, 50, 100]

fx = [reg.f_cdf(a,b)(x) for b in b_vals]
labels = [f"(a,b)=({a},{b})" for b in b_vals]
title = f"F-Distibution PDF for Range of Degrees of Freedom"
ylabel = r"$F(t;a,b)$"
xlabel = r"$t$"
plot = f"f_test_f_dist_cdf_b_scan_{a}"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.45, 0.4], [0.0, 1.1], title, plot)

# %%

ntrials = 10000
a = 100
b = 50
t = numpy.zeros(ntrials)

for i in range(ntrials):
    a_samples = reg.brownian_noise(1.0, a)
    b_samples = reg.brownian_noise(1.0, b)
    t[i] = (numpy.sum(a_samples**2)/(a-1))/(numpy.sum(b_samples**2)/(b-1))

# %%

ylabel = r"$f(t;100,50)$"
xlabel = r"$t$"
title = f"Sampled F-Distribution {ntrials} Trails, ({a-1},{b-1}) Degrees of Freedom"
pdf = reg.f_pdf(a-1, b-1)
reg.pdf_samples(pdf, t, title, ylabel, xlabel, "f_test_f_distribution_example_1_simualtion")

# %%

b1 = b-1
a1 = a-1
numpy.mean(t)
numpy.var(t)

b1/(b1-2)
2*b1**2*(b1+a1-2)/(a1*(b1-2)**2*(b1-4))

# %%

npts = 1001
xmax = 10.0
x = numpy.linspace(0.0001, xmax, npts)

a_data =[93, 50, 53, 92, 21, 1, 2, 85, 86, 22]
b_data = [12, 11, 20, 31, 65, 10, 3, 9, 1, 4, 12, 87, 43, 23, 52, 49, 17, 17, 14, 24]

a = len(a_data) - 1
b = len(b_data) - 1
acceptance_level = 0.95

title = f"F-Distibution CDF, Number of Degrees of Freedom: {a}, {b}"
ylabel = r"$F(t;9,19)$"
xlabel = r"$t$"
plot = f"f_test_example_1_cdf"
reg.hypothesis_region_plot(reg.f_cdf(a, b)(x), x, ylabel, xlabel, acceptance_level, title, plot)

# %%

a_var = numpy.var(a_data, ddof=1.0)
b_var = numpy.var(b_data, ddof=1.0)

t = a_var/b_var

t

reg.f_cdf(a, b)(2.88)
1.0-reg.f_cdf(a, b)(t)
