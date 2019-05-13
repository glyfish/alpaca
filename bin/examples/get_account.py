import os
import sys
import alpaca_trade_api

wd = os.getcwd()
sys.path.append(wd)

from lib import alpaca

credentials = alpaca.load_paper_credentials()

api = alpaca_trade_api.REST(key_id=credentials['key_id'], secret_key=credentials['secret_key'], base_url=credentials['base_url'])
account = api.get_account()
print(account)
