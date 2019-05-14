# %%
%load_ext autoreload
%autoreload 2

import os
import sys
from datetime import datetime
import backtrader
from matplotlib import pyplot
from lib import config
import pandas

pyplot.style.use(config.glyfish_style)
wd = os.getcwd()
yahoo_root = os.path.join(wd, 'data', 'yahoo')

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

data_file = os.path.join(yahoo_root, 'AAPL.csv')
data = backtrader.feeds.YahooFinanceCSVData(dataname=data_file, fromdate=datetime(2000, 1, 1), todate=datetime(2000, 12, 31))
cerebro.adddata(data)
cerebro.run()
cerebro.broker.get_value()

# %%

datframe = pandas.read_csv(data_file, parse_dates=True)

# %%

class TestStrategy(backtrader.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close

    def next(self):
        self.log('Close, %.2f' % self.dataclose[0])

cerebro.addstrategy(TestStrategy)

# %%

cerebro.broker.get_value()
result = cerebro.run()
result[0].datas[0].datetime.date(-251).isoformat()
cerebro.broker.get_value()
