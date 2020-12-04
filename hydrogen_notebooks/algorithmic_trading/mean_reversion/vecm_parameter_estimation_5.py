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
# Test one cointegration vector with one cointegration vector
example = 5
assumed_rank = 2
title_prefix = f"Trivariate VECM {assumed_rank} Cointegrating Vectors"

nsample = 1000
α = numpy.matrix([[-0.25, 0.0],
                  [0.0, -0.5],
                  [0.0, 0.0]])
β = numpy.matrix([[1.0, -0.1, -0.5],
                  [-0.25, 1.0, -0.25]])
a = numpy.matrix([[0.5, 0.0, 0.0],
                  [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5]])
Ω = numpy.matrix([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

title = title_prefix
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
plot = f"vecm_analysis_{example}_samples"
df = vecm.vecm_generate_sample(α, β, a, Ω, nsample)

# %%

vecm.comparison_plot(title, df, α.T, β, labels, [0.65, 0.1], plot)

# %%

vecm.sample_adf_test(df, report=True)

# %%

title = title_prefix + " First Difference"
labels = [r"$Δx_1$", r"$Δx_2$", r"$Δx_3$"]
plot = f"vecm_analysis_{example}_samples_diff_1"
df_diff_1 = vecm.difference(df)
vecm.comparison_plot(title, df_diff_1, α.T, β, labels, [0.1, 0.1], plot)

# %%

vecm.sample_adf_test(df_diff_1, report=True)

# %%

vecm.causality_matrix(df_diff_1, 1, cv=0.05)

# %%

rank = vecm.johansen_coint(df, report=True)

if rank != assumed_rank:
    print(f"Assumed rank {assumed_rank} not equal to estimated rank {rank}")

# %%

maxlag = vecm.aic_order(df_diff_1, 15)
print(f"AIC max lag: {maxlag}")

# %%

title = title_prefix + " ACF-PCF"
plot = f"vecm_analysis_{example}_acf_pcf"
max_time_lag = 9
vecm.acf_pcf_plot(title, df, max_time_lag, plot)

# %%

df.corr()

# %%

df.cov()

# %%

title = title_prefix + " Scatter Matrix"
plot = f"vecm_analysis_{example}_scatter_matrix"
vecm.scatter_matrix_plot(title, df, plot)

# %%

df_diff_1.corr()

# %%

df_diff_1.cov()

# %%

title = title_prefix + " Difference Scatter Matrix"
plot = f"vecm_analysis_{example}_differencescatter_matrix"
vecm.scatter_matrix_plot(title, df_diff_1, plot)

# %%

results = vecm.cointgration_params_estimate(df, rank)
for result in results:
    print(result.summary())

# %%

vecm_result = vecm.vecm_estimate(df, 1, rank, report=True)

# %%

vecm.residual_adf_test(df, vecm_result.beta.T, report=True)

# %%

train = vecm.vecm_train(df, 1, rank, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_analysis_{example}_x1_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Training"
plot = f"vecm_analysis_{example}_x2_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Training"
plot = f"vecm_analysis_{example}_x3_training"
vecm.training_plot(title, train, var, [0.7, 0.2], plot)

# %%

pred = vecm.vecm_prediction(df.columns, vecm_result, 10)

# %%

var = "x1"
title = title_prefix + r" $x_1$ Training"
plot = f"vecm_analysis_{example}_x1_training"
lag = 30
vecm.prediction_plot(title, df, pred, lag, var, plot)

# %%

var = "x2"
title = title_prefix + r" $x_2$ Prediction"
plot = f"vecm_analysis_{example}_x2_prediction"
lag = 20
vecm.prediction_plot(title, df, pred, lag, var, plot)

# %%

var = "x3"
title = title_prefix + r" $x_3$ Prediction"
plot = f"vecm_analysis_{example}_x3_prediction"
lag = 20
vecm.prediction_plot(title, df, pred, lag, var, plot)
