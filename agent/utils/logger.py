# Logging functionality

import logging

def setup_logger(logger_name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Sets up a logger with the given name and log file."""
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s : %(message)s")
    fileHandler = logging.FileHandler(log_file, mode="w")
    fileHandler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

    return l