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

    params = (
        ('exitbars', 5),
        ('stake', 10),
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sizer.setsizing(self.params.stake)
        logger.info(f"Data source: {self.datas[0].p.dataname}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' % (order.executed.price, order.executed.value, order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' % (order.executed.price, order.executed.value, order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin. order.Rejected]:
            self.log('ORDER Cancelled/Margin/Rejected')

        self.order = None

    def next(self):
        self.log('Close: %.2f' % (self.dataclose[0]))

        # Do nothing if order is pending
        if self.order:
            return

        # if not in market buy if criterion is met
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1] and self.dataclose[-1] < self.dataclose[-2]:
                self.log('CREATE BUY ORDER: %.2f, %.2f, %.2f' % (self.dataclose[0], self.dataclose[-1], self.dataclose[-2]))
                self.order = self.buy()

        # if in market sell if criterion is met
        else:
            # sell if positions has been held for more than hold_barsd
            if len(self) >= (self.bar_executed + self.params.exitbars):
                self.log('CREATE SELL ORDER: %.2f' % self.dataclose[0])
                self.order = self.sell()

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

# main
def main():

    # get data
    data_file = os.path.join(yahoo_root, 'AAPL.csv')
    data = backtrader.feeds.YahooFinanceCSVData(dataname=data_file, fromdate=datetime(2010, 1, 1), todate=datetime(2010, 12, 31))
    cerebro = backtrader.Cerebro()
    cerebro.adddata(data)

    # add strategy
    cerebro.addstrategy(TestStrategy)

    # configure broker
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    # run strategy
    logger.info(f"Inital Value: {cerebro.broker.get_value()}")
    cerebro.run()
    logger.info(f"Final Value: {cerebro.broker.get_value()}")

# run algorithm
if __name__ == '__main__':
    main()
