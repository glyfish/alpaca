import os
import sys

wd = os.getcwd()
sys.path.append(wd)

from lib import alpaca

credentials = alpaca.load_paper_credentials()
