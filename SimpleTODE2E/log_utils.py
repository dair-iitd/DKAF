import os
import socket
import logging


def setup_logging(cfg):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    os.makedirs(cfg['destpath'], exist_ok=True)
    hostname = socket.gethostname()
    local_rank = os.environ.get('DDP_RANK', '0')
    tag = f"{hostname}_rank_{local_rank}"

    ch = logging.StreamHandler()
    fname = os.path.join(cfg['destpath'], f'run_{tag}.log')
    fh = logging.FileHandler(fname)
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)

    format_line = '%(asctime)s %(module)s:%(funcName)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_line)
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
