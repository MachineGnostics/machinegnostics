import logging

def get_logger(name: str, level: int = logging.WARNING) -> logging.Logger:
    """
    Create and configure a logger with the given name and level.

    Args:
        name (str): Name of the logger, typically `__name__`.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger