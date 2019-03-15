import logging
import os


def get_logger(name, job_dir='.'):
    """ Returns logger that prints on stdout at INFO level and on file at DEBUG level. """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        # stream handler ensures that logging events are passed to stdout
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # file handler ensures that logging events are passed to log file
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)

        fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
