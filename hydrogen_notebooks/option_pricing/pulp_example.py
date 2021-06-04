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
# Maximize: z = x + 2y
# Constraints: 2x + y <= 20, -4x + 5y <= 10, -x + 2y >= -2, -x + 5y = 15, x >= 0, y >= 0

# %%

model = pulp.LpProblem(name="sub-replicating-portfolio", sense=pulp.LpMaximize)
x = pulp.LpVariable(name="x", lowBound=0)
y = pulp.LpVariable(name="y", lowBound=0)

# Constraints
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Objective Function
model += pulp.lpSum([x, 2 * y])

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
