# %%

%load_ext autoreload
%autoreload 2

import os
import sys
import numpy
from matplotlib import pyplot
from lib import config
import pulp

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
pyplot.style.use(config.glyfish_style)

# Example From https://realpython.com/linear-programming-python/
# Maximize: Vxy(0) = 30x + y
# Constraints: 36x + 1.05y >= 4, 33x + 1.05y >= 1, 27x + 1.05y >= 0

# %%

model = pulp.LpProblem(name="sub-replicating-portfolio", sense=pulp.LpMaximize)
x = pulp.LpVariable(name="x")
y = pulp.LpVariable(name="y")

# Constraints
model += (36 * x + 1.05 * y <= 4, "up_constraint")
model += (33 * x + 1.05 * y <= 1, "mid_constraint")
model += (27 * x + 1.05 * y <= 0, "down_constraint")

# Objective Function
model += pulp.lpSum([30 * x, y])

model

# %%

status = model.solve()

# %%

print(f"status: {model.status}, {pulp.LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")
