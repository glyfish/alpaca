import os
import yaml
import io

def load_paper_credentials():
    credentials_file = os.path.join(os.environ["HOME"], ".alpaca/paper_trading_credentials.yml")
    with io.open(config_file, "r") as stream:
        credentials = yaml.safe_load(stream)
    return credentials
