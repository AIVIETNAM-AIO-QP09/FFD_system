"""Module for handling logging configuration across the pipeline."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a centralized logger for the pipeline.
    Ensures that logging formatting and levels are standard across the project.
    """
    logger = logging.getLogger(name)

    # Only configure the logger if it hasn't been configured yet
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler and set level to INFO
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add formatter to ch
        ch.setFormatter(formatter)

        # Add ch to logger
        logger.addHandler(ch)

        # Prevent log propagation to the root logger
        logger.propagate = False

    return logger
