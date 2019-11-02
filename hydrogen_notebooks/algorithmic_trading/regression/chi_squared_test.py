# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import special
from scipy.stats import chi2
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# %%

npts = 1001
xmax = 10.0
x = numpy.linspace(0.0001, xmax, npts)

# %%

k = 3
title = r"$\chi^2$"+f" PDF from scipy, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(chi2.pdf(x, k), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_3_pdf_scipi")

# %%

k = 3
title = r"$\chi^2$"+f" CDF from scipy, Number of Degrees of Freedom: {k}"
ylabel = r"$F(\chi^2;3)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(chi2.cdf(x, k), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_3_cdf_scipi")

# %%

k = 3
title = r"$\chi^2$"+f" PDF, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;3)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(reg.chi_squared_pdf(k)(x), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_3_pdf")

# %%

k = 3
title = r"$\chi^2$"+f" CDF, Number of Degrees of Freedom: {k}"
ylabel = r"$F(\chi^2;3)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(reg.chi_squared_cdf(k)(x), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_3_cdf")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_pdf(k)(x) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ PDF for Range of Degrees of Freedom"
ylabel = r"$f(\chi^2;k)$"
xlabel = r"$\chi^2$"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.8, 0.8], [0.0, 0.5], title, "chi_squared_test_chi_squared_pdf_scan")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_cdf(k)(x) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ CDF for Range of Degrees of Freedom"
ylabel = r"$F(\chi^2;k)$"
xlabel = r"$\chi^2$"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.68, 0.3], [0.0, 1.0], title, "chi_squared_test_chi_squared_cdf_scan")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_tail(k)(x) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ Tail CDF for Range of Degrees of Freedom"
ylabel = r"$1-F(\chi^2;k)$"
xlabel = r"$\chi^2$"
reg.distribution_multiplot(fx, x, labels, ylabel, xlabel, [0.33, 0.4], [0.0, 1.0], title, "chi_squared_test_chi_squared_tail_cdf_scan")

# %%

k = 2
title = r"$\chi^2$"+f" PDF, Number of Degrees of Freedom: {k}"
ylabel = r"$f(\chi^2;2)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(reg.chi_squared_pdf(k)(x), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_example_1_pdf")

# %%

k = 2
title = r"$\chi^2$"+f" CDF, Number of Degrees of Freedom: {k}"
ylabel = r"$F(\chi^2;2)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(reg.chi_squared_cdf(k)(x), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_example_1_cdf")

# %%

k = 2
title = r"$\chi^2$"+f" CDF, Number of Degrees of Freedom: {k}"
ylabel = r"1-$F(x\chi^2;2)$"
xlabel = r"$\chi^2$"
reg.distribution_plot(reg.chi_squared_tail(k)(x), x, title, ylabel, xlabel, "chi_squared_test_chi_squared_example_1_tail_cdf")

# %%

acceptance_level = 0.05
k = 2
title = r"$\chi^2$"+f" Tail CDF, Number of Degrees of Freedom: {k}"
ylabel = "Probability"
xlabel = r"$\chi^2$"
reg.hypothesis_region_plot(reg.chi_squared_tail(k)(x), x, ylabel, xlabel, acceptance_level, title, "chi_squared_test_chi_squared_example_1_hypothesis_region")

# %%

k = 2
ntrials = 10000
nsample = 1000
chi = numpy.zeros(ntrials)
p = [0.18, 0.50, 0.32]

for i in range(ntrials):
    result = numpy.random.multinomial(nsample, p)
    for j in range(len(p)):
        chi[i] += (result[j] - nsample*p[j])**2/(nsample*p[j])

# %%

ylabel = r"$f(x\chi^2;2)$"
xlabel = r"$\chi^2$"
title = r"$\chi^2$"+f" Sampled Distribution {ntrials} Trails, {k} Degrees of Freedom"
pdf = reg.chi_squared_pdf(k)
reg.pdf_samples(pdf, chi, title, ylabel, xlabel, "chi_squared_test_chi_squared_example_1_simualtion")

# %%

k = 2
ntrials = 1000
nsample = 1000
chi = numpy.zeros(ntrials)
p = [0.18, 0.50, 0.32]

for i in range(ntrials):
    result = numpy.random.multinomial(nsample, p)
    for j in range(len(p)):
        chi[i] += (result[j] - nsample*p[j])**2/(nsample*p[j])

# %%

ylabel = r"$f(x\chi^2;2)$"
xlabel = r"$\chi^2$"
title = r"$\chi^2$"+f" Sampled Distribution {ntrials} Trails, {k} Degrees of Freedom"
pdf = reg.chi_squared_pdf(k)
reg.pdf_samples(pdf, chi, title, ylabel, xlabel, "chi_squared_test_chi_squared_example_1_simualtion_2")
