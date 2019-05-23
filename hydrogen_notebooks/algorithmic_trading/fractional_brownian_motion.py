# %%
%load_ext autoreload
%autoreload 2

import os
import sys
from lib import brownian_motion
import numpy

wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')
