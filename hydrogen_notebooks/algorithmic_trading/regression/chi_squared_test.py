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
title = r"$\chi^2$"+f" PDF from scipy, Number of Degress of Freedom: {k}"
ylabel = r"$f(x;3)$"
reg.distribution_plot(chi2.pdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_3_pdf_scipi")

# %%

k = 3
title = r"$\chi^2$"+f" CDF from scipy, Number of Degress of Freedom: {k}"
ylabel = r"$F(x;3)$"
reg.distribution_plot(chi2.cdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_3_cdf_scipi")

# %%

k = 3
title = r"$\chi^2$"+f" PDF, Number of Degress of Freedom: {k}"
ylabel = r"$f(x;3)$"
reg.distribution_plot(reg.chi_squared_pdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_3_pdf")

# %%

k = 3
title = r"$\chi^2$"+f" CDF, Number of Degress of Freedom: {k}"
ylabel = r"$F(x;3)$"
reg.distribution_plot(reg.chi_squared_cdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_3_cdf")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_pdf(x, k) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ PDF for Range of Degress of Freedom"
ylabel = r"$f(x;k)$"
reg.distribution_multiplot(fx, x, labels, ylabel, [0.8, 0.8], [0.0, 0.5], title, "chi_squared_test_chi_squared_pdf_scan")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_cdf(x, k) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ CDF for Range of Degress of Freedom"
ylabel = r"$F(x;k)$"
reg.distribution_multiplot(fx, x, labels, ylabel, [0.68, 0.3], [0.0, 1.0], title, "chi_squared_test_chi_squared_cdf_scan")

# %%

k_vals = [1, 2, 3, 4, 6, 9, 12]
fx = [reg.chi_squared_tail(x, k) for k in k_vals]
labels = [f"k={k}" for k in k_vals]
title = r"$\chi^2$ CDF for Range of Degress of Freedom"
ylabel = r"$1-F(x;k)$"
reg.distribution_multiplot(fx, x, labels, ylabel, [0.33, 0.4], [0.0, 1.0], title, "chi_squared_test_chi_squared_tail_cdf_scan")

# %%

k = 2
title = r"$\chi^2$"+f" PDF, Number of Degress of Freedom: {k}"
ylabel = r"$f(x;2)$"
reg.distribution_plot(reg.chi_squared_pdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_example_1_pdf")

# %%

k = 2
title = r"$\chi^2$"+f" CDF, Number of Degress of Freedom: {k}"
ylabel = r"$F(x;2)$"
reg.distribution_plot(reg.chi_squared_cdf(x, k), x, title, ylabel, "chi_squared_test_chi_squared_example_1_cdf")

# %%

k = 2
title = r"$\chi^2$"+f" CDF, Number of Degress of Freedom: {k}"
ylabel = r"1-$F(x;2)$"
reg.distribution_plot(reg.chi_squared_tail(x, k), x, title, ylabel, "chi_squared_test_chi_squared_example_1_tail_cdf")

# %%

acceptance_level = 0.05
k = 2
title = r"$\chi^2$"+f" Tail CDF, Number of Degress of Freedom: {k}"
ylabel = r"1-$F(x;2)$"
reg.hypothesis_region_plot(reg.chi_squared_tail(x, k), x, acceptance_level, title, ylabel, "chi_squared_test_chi_squared_example_1_hypothesis_region")
