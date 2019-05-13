from datetime import timedelta, datetime
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

def setup_logging(file_name=None, max_bytes=200000000, backup_count=5):
    logger = logging.getLogger("cryptocoins")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('(%(levelname)s|%(asctime)s|%(module)s|%(filename)s, %(lineno)s) %(message)s')
    if file_name is not None:
        handler = RotatingFileHandler(file_name, "a", max_bytes, backup_count)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info(f"LOGGING to {file_name}")
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("LOGGING to stdout")
    return logger
