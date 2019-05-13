from datetime import datetime
import backtrader

class SmaCross(backtrader.SignalStrategy):
    params = (('pfast', 10), ('pslow', 30),)
    def __init__(self):
        sma1, sma2 = backtrader.ind.SMA(period=self.p.pfast), backtrader.ind.SMA(period=self.p.pslow)
        self.signal_add(backtrader.SIGNAL_LONG, backtrader.ind.CrossOver(sma1, sma2))

cerebro = backtrader.Cerebro()
cerebro.broker.setcash(100000)
cerebro.broker.setcommission(commission=0.0)
cerebro.addsizer(backtrader.sizers.PercentSizer, percents=20)

data = backtrader.feeds.YahooFinanceData(dataname='MSFT', fromdate=datetime(2011, 1, 1), todate=datetime(2012, 12, 31))
cerebro.adddata(data)

cerebro.addstrategy(SmaCross)
cerebro.run()
cerebro.plot()
