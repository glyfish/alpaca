import os
import sys
import alpaca_trade_api
import pandas
import time

wd = os.getcwd()
sys.path.append(wd)

from lib import alpaca
from lib.utils import setup_logging

credentials = alpaca.load_paper_credentials()

api = alpaca_trade_api.REST(key_id=credentials['key_id'], secret_key=credentials['secret_key'], base_url=credentials['base_url'])

# setup logging
log_out = sys.argv[1] if len(sys.argv) > 1 else '/var/log/apps/alpaca/paper_trade_example.log'

if log_out == 'stdout':
    logger = setup_logging()
else:
    logger = setup_logging(file_name=log_out)


def _get_prices(symbols, end_dt, max_workers=5):
    # Get the map of DataFrame price data from Alpaca's data API.
    start_dt = end_dt - pandas.Timedelta('50 days')
    start = start_dt.strftime('%Y-%m-%d')
    end = end_dt.strftime('%Y-%m-%d')

    def get_barset(symbols):
        return api.get_barset(symbols, 'day', limit = 50, start=start, end=end)

    # The maximum number of symbols we can request at once is 200.
    barset = None
    idx = 0
    while idx <= len(symbols) - 1:
        if barset is None:
            barset = get_barset(symbols[idx:idx+200])
        else:
            barset.update(get_barset(symbols[idx:idx+200]))
        idx += 200

    return barset.df

# Get the map of prices in DataFrame with the symbol name key.
def prices(symbols):
    now = pandas.Timestamp.now(tz='America/New_York')
    end_dt = now
    if now.time() >= pandas.Timestamp('09:30', tz='America/New_York').time():
        end_dt = now - pandas.Timedelta(now.strftime('%H:%M:%S')) - pandas.Timedelta('1 minute')
    return _get_prices(symbols, end_dt)

# Calculate scores based on the indicator and return the sorted result.
def calc_scores(price_df, dayindex=-1):
    diffs = {}
    param = 10
    for symbol in price_df.columns.levels[0]:
        df = price_df[symbol]
        if len(df.close.values) <= param:
            continue
        ema = df.close.ewm(span=param).mean()[dayindex]
        last = df.close.values[dayindex]
        diff = (last - ema) / last
        diffs[symbol] = diff

    return sorted(diffs.items(), key=lambda x: x[1])

def get_orders(api, price_df, position_size=100, max_positions=5):
    ranked = calc_scores(price_df)
    to_buy = set()
    to_sell = set()
    account = api.get_account()
    # take the top one twentieth out of ranking,
    # excluding stocks too expensive to buy a share
    for symbol, _ in ranked[:len(ranked) // 20]:
        price = float(price_df[symbol].close.values[-1])
        if price > float(account.cash):
            continue
        to_buy.add(symbol)

    # now get the current positions and see what to buy,
    # what to sell to transition to today's desired portfolio.
    positions = api.list_positions()
    logger.info(positions)
    holdings = {p.symbol: p for p in positions}
    holding_symbol = set(holdings.keys())
    to_sell = holding_symbol - to_buy
    to_buy = to_buy - holding_symbol
    orders = []

    # if a stock is in the portfolio, and not in the desired
    # portfolio, sell it
    for symbol in to_sell:
        shares = holdings[symbol].qty
        orders.append({'symbol': symbol, 'qty': shares,'side': 'sell'})
        logger.info(f'order(sell): {symbol} for {shares}')

    # likewise, if the portfoio is missing stocks from the
    # desired portfolio, buy them. We sent a limit for the total
    # position size so that we don't end up holding too many positions.
    max_to_buy = max_positions - (len(positions) - len(to_sell))
    for symbol in to_buy:
        if max_to_buy <= 0:
            break
        shares = position_size // float(price_df[symbol].close.values[-1])
        if shares == 0.0:
            continue
        orders.append({'symbol': symbol, 'qty': shares, 'side': 'buy'})
        logger.info(f'order(buy): {symbol} for {shares}')
        max_to_buy -= 1
    return orders

def trade(orders, wait=30):
    # This is where we actually submit the orders and wait for them to fill.
    # Waiting is an important step since the orders aren't filled automatically,
    # which means if your buys happen to come before your sells have filled,
    # the buy orders will be bounced. In order to make the transition smooth,
    # we sell first and wait for all the sell orders to fill before submitting
    # our buy orders.

    # process the sell orders first
    sells = [o for o in orders if o['side'] == 'sell']
    for order in sells:
        try:
            logger.info(f'submit(sell): {order}')
            api.submit_order(symbol=order['symbol'], qty=order['qty'], side='sell', type='market', time_in_force='day')
        except Exception as e:
            logger.error(e)
    count = wait
    while count > 0:
        pending = api.list_orders()
        if len(pending) == 0:
            logger.info(f'all sell orders done')
            break
        logger.info(f'{len(pending)} sell orders pending...')
        time.sleep(1)
        count -= 1

    # process the buy orders next
    buys = [o for o in orders if o['side'] == 'buy']
    for order in buys:
        try:
            logger.info(f'submit(buy): {order}')
            api.submit_order(
                symbol=order['symbol'],
                qty=order['qty'],
                side='buy',
                type='market',
                time_in_force='day',
            )
        except Exception as e:
            logger.error(e)
    count = wait
    while count > 0:
        pending = api.list_orders()
        if len(pending) == 0:
            logger.info(f'all buy orders done')
            break
        logger.info(f'{len(pending)} buy orders pending...')
        time.sleep(1)
        count -= 1

def main():
    done = None
    logger.info('start running')
    while True:
        # clock API returns the server time including
        # the boolean flag for market open
        clock = api.get_clock()
        now = clock.timestamp
        if clock.is_open and done != now.strftime('%Y-%m-%d'):

            price_df = prices(alpaca.SP500)
            orders = get_orders(api, price_df)
            trade(orders)

            # flag it as done so it doesn't work again for the day
            # TODO: this isn't tolerant to process restarts, so this
            # flag should probably be saved on disk
            done = now.strftime('%Y-%m-%d')
            logger.info(f'done for {done}')

        time.sleep(1)

if __name__ == '__main__':
    main()
