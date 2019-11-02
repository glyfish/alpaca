# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from scipy import special
from scipy.stats import t as student_t
from matplotlib import pyplot
from lib import config
from lib import regression as reg

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

 # %%

npts = 1001
xmax = 6.0
x = numpy.linspace(-xmax, xmax, npts)

# %%

n = 3
title = f"Student-t PDF from scipy, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(student_t.pdf(x, n), x, title, ylabel, xlabel, "student_t_test_3_pdf_scipi")

# %%

n = 3
title = f"Student-t CDF from scipy, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(student_t.cdf(x, n), x, title, ylabel, xlabel, "student_t_test_3_cdf_scipi")

# %%

n = 3
title = f"Student-t PDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
reg.distribution_plot(reg.student_t_pdf(n)(x), x, title, ylabel, xlabel, "student_t_test_3_pdf")

# %%

n = 3
title = f"Student-t CDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_cdf(n)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_3_cdf")

# %%

n = 3
title = f"Student-t Tails CDF, Number of Degrees of Freedom: {n}"
ylabel = r"$f(t;3)$"
xlabel = r"$t$"
cdf = [reg.student_t_tail(n)(i) for i in x]
reg.distribution_plot(cdf, x, title, ylabel, xlabel, "student_t_test_3_tail_cdf")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t PDF, for Range of Degrees of Freedom Compared to Unit Normal"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
pdf = [reg.student_t_pdf(n)(x) for n in n_vals]
reg.distribution_comparission_multiplot(pdf, reg.normal()(x), x, labels, ylabel, xlabel, [0.33, 0.8], [0.0, 0.42], title, "student_t_test_pdf_scan")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t CDF, for Range of Degrees of Freedom"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
cdf = []
for n in n_vals:
    cdf.append([reg.student_t_cdf(n)(i) for i in x])

reg.distribution_multiplot(cdf, x, labels, ylabel, xlabel, [0.33, 0.8], [0.0, 1.05], title, "student_t_test_cdf_scan")

# %%

n_vals = [1, 3, 5, 8, 12]
title = f"Student-t Tail CDF, for Range of Degrees of Freedom Compared to Unit Normal"
labels = [f"n={n}" for n in n_vals]
labels.append("normal")
ylabel = r"$f(t;n)$"
xlabel = r"$t$"
cdf = []
for n in n_vals:
    cdf.append([reg.student_t_tail(n)(i) for i in x])

reg.distribution_multiplot(cdf, x, labels, ylabel, xlabel, [0.33, 0.5], [0.0, 1.05], title, "student_t_test_tail_cdf_scan")


# %%

n = 3
μ = 5.0
σ = 4.0
nsamples = 100
ntrials = 1000
t = numpy.zeros(ntrials)

for k in range(ntrials):
    samples = numpy.zeros(nsamples)
    for i in range(nsamples):
        for j in range(n):
            samples[i] += numpy.random.normal(μ, σ)
    t[k] = (numpy.mean(samples) - μ)/(numpy.sqrt(numpy.var(samples)/n))

# %%

ylabel = r"$f(t;3)$"
xlabel = r"$t$"
title = f"t-Test Sampled Distribution {ntrials} Trails, {k} Degrees of Freedom"
reg.pdf_samples(reg.student_t_pdf(n), t, title, ylabel, xlabel, "t_test_example_0_simulation")
