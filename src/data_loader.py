"""Module for data ingestion and configuration loading."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads system configuration parameters from a YAML file."""
    # Resolve path relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    full_path = project_root / config_path

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration successfully loaded from: %s", full_path)
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found at path: %s", full_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML configuration: %s", str(e))
        raise


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Reads the raw financial transaction dataset from disk efficiently."""
    project_root = Path(__file__).resolve().parent.parent
    full_path = project_root / file_path

    logger.info("Initiating data retrieval from: %s", full_path)
    try:
        df = pd.read_csv(full_path)
        logger.info("Data ingested successfully. Shape: %s", df.shape)
        return df
    except FileNotFoundError:
        logger.error("Target data file not found at path: %s", full_path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("Target data file is empty: %s", full_path)
        raise
