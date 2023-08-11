import logging
import sys


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    bold_red = "\x1b[1;31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)-s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class ImmediateStreamHandler(logging.StreamHandler):
    def emit(self, record):
        if sys.stdout is sys.__stdout__:
            super().emit(record)
            self.flush()
        else:
            print(self.format(record),flush=True)
            self.flush()


# Create a root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a console handler
ch = ImmediateStreamHandler()
ch.setLevel(logging.INFO)

# Add the formatter to the console handler and logger
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
