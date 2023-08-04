import logging


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


# Create a root logger
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Add the formatter to the console handler and logger
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
