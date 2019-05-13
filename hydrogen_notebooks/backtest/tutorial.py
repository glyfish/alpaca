# %%
%load_ext autoreload
%autoreload 2

from datetime import datetime
import backtrader
from matplotlib import pyplot
from lib import config

pyplot.style.use(config.glyfish_style)

# %%
# Initialize simulation engine and set initail broker balances
cerebro = backtrader.Cerebro()
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.0)

# %%

cerebro.broker.get_value()
cerebro.run()
cerebro.broker.get_value()

# %%

data = backtrader.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2011, 1, 1), todate=datetime(2012, 12, 31))
data.p.t
