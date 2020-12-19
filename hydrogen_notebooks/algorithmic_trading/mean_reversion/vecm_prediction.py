# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
import pandas
from matplotlib import pyplot

from lib import config
from lib import vecm

pyplot.style.use(config.glyfish_style)

# %%
# Test one cointegration vector
nsample = 1000
α = numpy.matrix([-0.2, 0.0, 0.0]).T
β = numpy.matrix([1.0, -0.25, -0.5])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

df = vecm.vecm_generate_sample(α, β, a, Ω, nsample)

# %%

example = 1
rank = 1
maxlags = 1
title_prefix = f"Trivariate VECM: Rank={rank}, Maxlags={maxlags}, "

# %%

title = title_prefix
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = f"vecm_prediction_{example}_samples"

vecm.comparison_plot(title, df, α.T, β, labels, [0.45, 0.075], plot)

# %%

vecm_result = vecm.vecm_estimate(df, maxlags, rank, report=True)

# %%

train = vecm.vecm_train(df, maxlags, rank, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_prediction_{example}_x1_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Training"
plot = f"vecm_prediction_{example}_x2_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Training"
plot = f"vecm_prediction_{example}_x3_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

example = 2
rank = 2
maxlags = 1
title_prefix = f"Trivariate VECM: Rank={rank}, Maxlags={maxlags}, "

# %%

vecm_result = vecm.vecm_estimate(df, maxlags, rank, report=True)

# %%

train = vecm.vecm_train(df, maxlags, rank, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_prediction_{example}_x1_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Training"
plot = f"vecm_prediction_{example}_x2_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Training"
plot = f"vecm_prediction_{example}_x3_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

example = 3
rank = 1
maxlags = 2
title_prefix = f"Trivariate VECM: Rank={rank}, Maxlags={maxlags}, "

# %%

vecm_result = vecm.vecm_estimate(df, maxlags, rank, report=True)

# %%

train = vecm.vecm_train(df, maxlags, rank, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_prediction_{example}_x1_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Training"
plot = f"vecm_prediction_{example}_x2_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Training"
plot = f"vecm_prediction_{example}_x3_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

example = 4
rank = 2
maxlags = 2
title_prefix = f"Trivariate VECM: Rank={rank}, Maxlags={maxlags}, "

# %%

vecm_result = vecm.vecm_estimate(df, maxlags, rank, report=True)

# %%

train = vecm.vecm_train(df, maxlags, rank, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_prediction_{example}_x1_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Training"
plot = f"vecm_prediction_{example}_x2_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Training"
plot = f"vecm_prediction_{example}_x3_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)
