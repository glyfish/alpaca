import os
import sys
import argparse
from datetime import datetime
import backtrader
import pandas

wd = os.getcwd()
sys.path.append(wd)

from lib import alpaca
from lib.utils import setup_logging

# setup logging
logger = setup_logging()

# moving average strategy
class MovingAverageStretegy(backtrader.Strategy):

    params = dict(
        day_period=15,
        week_period=5
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        sma_day = backtrader.indicators.SMA(self.data0, period=self.p.day_period)
        sma_week = backtrader.indicators.SMA(self.data1, period=self.p.week_period)
        self.buysig = sma_day > sma_week()

    def next(self):
        self.log('Close: %.2f, %.2f, %s' % (self.data.close[0], self.data.close[1], self.buysig[0]))
        if self.buysig[0]:
            self.log('Buy %.2f' % (self.data))

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='MultiData Strategy')
    parser.add_argument('--symbol', '-s',
                        default='AAPL',
                        help='Data source symbol')
    return parser.parse_args()


# run strategy
def main():
    args = parse_args()
    symbol = args.symbol
    yahoo_root = os.path.join(wd, 'data', 'yahoo')
    data_file = os.path.join(yahoo_root, f"{symbol}.csv")
    logger.info(f"SYMBOL: {symbol}, {data_file}")

    daily_data = backtrader.feeds.YahooFinanceCSVData(dataname=data_file,
                                                      fromdate=datetime(2010, 1, 1),
                                                      todate=datetime(2010, 12, 31),
                                                      timeframe=backtrader.TimeFrame.Days)
    weekly_data = backtrader.feeds.YahooFinanceCSVData(dataname=data_file,
                                                       fromdate=datetime(2010, 1, 1),
                                                       todate=datetime(2010, 12, 31),
                                                       timeframe=backtrader.TimeFrame.Weeks)

    # configure cerebro
    cerebro = backtrader.Cerebro()
    cerebro.adddata(daily_data)
    cerebro.adddata(weekly_data)

    # add strategy
    cerebro.addstrategy(MovingAverageStretegy, day_period=15, week_period=5)

    # configure broker
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=0.0)

    # run strategy
    logger.info(f"Inital Value: {cerebro.broker.get_value()}")
    cerebro.run()
    logger.info(f"Final Value: {cerebro.broker.get_value()}")

if __name__ == '__main__':
    main()
