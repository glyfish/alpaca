import os
import sys
from datetime import datetime
import backtrader
import pandas

wd = os.getcwd()
sys.path.append(wd)
yahoo_root = os.path.join(wd, 'data', 'yahoo')

from lib import alpaca
from lib.utils import setup_logging

# setup logging
logger = setup_logging()

# Example strategy
class TestStrategy(backtrader.Strategy):

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        logger.info(f"Data source: {self.datas[0].p.dataname}")

    def next(self):
        self.log('Close: %.2f, Open: %.2f, Volume: %d' % (self.dataclose[0], self.datas[0].open[0], self.datas[0].volume[0]))
        if self.dataclose[0] < self.dataclose[-1] and self.dataclose[-1] < self.dataclose[-2]:
            self.log('CREATE BUY ORDER: %.2f, %.2f, %.2f' % (self.dataclose[0], self.dataclose[-1], self.dataclose[-2]))
            self.buy()

# main
def main():
    data_file = os.path.join(yahoo_root, 'AAPL.csv')
    data = backtrader.feeds.YahooFinanceCSVData(dataname=data_file, fromdate=datetime(2000, 1, 1), todate=datetime(2000, 12, 31))
    cerebro = backtrader.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.0)

    logger.info(f"Inital Value: ${cerebro.broker.get_value()}")
    cerebro.run()
    logger.info(f"Final Value: ${cerebro.broker.get_value()}")

# run algorithm
if __name__ == '__main__':
    main()
