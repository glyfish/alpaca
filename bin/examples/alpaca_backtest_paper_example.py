import os
import sys
import backtrader as bt
import alpaca_backtrader_api
import pandas

wd = os.getcwd()
sys.path.append(wd)

from lib import alpaca

credentials = alpaca.load_paper_credentials()

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

store = alpaca_backtrader_api.AlpacaStore(key_id=credentials['key_id'], secret_key=credentials['secret_key'], paper=True)

cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.0)
cerebro.addsizer(bt.sizers.PercentSizer, percents=20)

DataFactory = store.getdata
data0 = DataFactory(dataname='AAPL', timeframe=bt.TimeFrame.TFrame("Days"))
cerebro.adddata(data0)

cerebro.run(exactbars=1)
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.plot()
