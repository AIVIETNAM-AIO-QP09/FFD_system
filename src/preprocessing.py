"""
Module for data preprocessing, including logical anomaly fixes, missing data imputation,
and temporal feature extraction.
"""

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def fix_logical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects negative logical values inside system-generated behavioral features."""
    if "time_since_last_transaction" in df.columns:
        logger.info(
            "Preprocessing: Applying absolute values to clear "
            "negative 'time_since_last_transaction' values."
        )
        # Modify in place to save memory
        df["time_since_last_transaction"] = df["time_since_last_transaction"].abs()
    return df


def impute_conditional_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fills conditional missing fields in fraud metadata for legitimate users."""
    if "fraud_type" in df.columns:
        logger.info(
            "Preprocessing: Imputing missing values in 'fraud_type' "
            "with 'Legitimate' status."
        )
        df["fraud_type"] = df["fraud_type"].fillna("Legitimate")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parses and engineer meaningful temporal sub-components from raw transaction timestamps."""
    if "timestamp" in df.columns:
        logger.info(
            "Feature Engineering: Transforming 'timestamp' "
            "into Cyclical Temporal features."
        )
        try:
            datetime_series = pd.to_datetime(df["timestamp"], format="ISO8601")
            df["transaction_hour"] = datetime_series.dt.hour
            df["transaction_day_of_week"] = datetime_series.dt.dayofweek
            df["is_weekend"] = df["transaction_day_of_week"].isin([5, 6]).astype(int)
        except ValueError as e:
            logger.error("ValueError parsing timestamps: %s", str(e))
            raise
    return df


def transform_numerical_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """Applies log transformations to heavily right-skewed transaction financial values."""
    if "amount" in df.columns:
        logger.info(
            "Feature Engineering: Applying log1p transformation "
            "to heavily skewed 'amount' distribution."
        )
        df["amount_log"] = np.log1p(df["amount"])
    return df


def run_baseline_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Main execution sequence consolidating all structural baseline cleaning and raw feature engineering."""
    logger.info("Executing Enterprise Baseline Preprocessing Pipeline...")

    # Create a single copy at the beginning of the pipeline if necessary to preserve original data
    # Or in our case, just mutate the passed df directly if the caller doesn't need the raw version.
    # We will mutate the dataframe to optimize for memory.
    df_processed = df

    df_processed = fix_logical_anomalies(df_processed)
    df_processed = impute_conditional_missing_data(df_processed)
    df_processed = extract_temporal_features(df_processed)
    df_processed = transform_numerical_distributions(df_processed)

    logger.info(
        "Enterprise Baseline Preprocessing Pipeline execution " "finished successfully."
    )
    return df_processed
