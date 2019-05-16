import os
import sys
import argparse
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

# moving average strategy
class MovingAverageStretegy(backtrader.Strategy):

    params = dict(
        fast_period=30.0,
        slow_period=50.0
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        logger.info('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.movav = backtrader.SimpleMovingAverage(self.data, period=self.p.fast_period)

    def next(self):
        if self.mvavg.lines.sma[0] > self.data.lines.close[0]:
            self.log('SMA: %.2f, Close: %.2f' % (self.mvavg.lines.sma[0], self.data.lines.close[0]))

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
    print(args.symbol)

if __name__ == '__main__':
    main()
