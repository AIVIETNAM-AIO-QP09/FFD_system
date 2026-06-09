import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    extract_temporal_features,
    fix_logical_anomalies,
    impute_conditional_missing_data,
    transform_numerical_distributions,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "time_since_last_transaction": [-10, 5, -3, 0],
            "fraud_type": ["Card", np.nan, "Identity", np.nan],
            "timestamp": [
                "2023-07-06T21:04:01",
                "2023-07-07T10:00:00",
                "2023-07-08T08:30:00.123",
                "2023-07-09T14:45:00",
            ],
            "amount": [100.0, 50.0, 0.0, 1000.0],
        }
    )


def test_fix_logical_anomalies(sample_data):
    df_result = fix_logical_anomalies(sample_data)
    assert df_result["time_since_last_transaction"].min() >= 0
    assert df_result["time_since_last_transaction"].tolist() == [10, 5, 3, 0]


def test_impute_conditional_missing_data(sample_data):
    df_result = impute_conditional_missing_data(sample_data)
    assert not df_result["fraud_type"].isna().any()
    assert df_result["fraud_type"].tolist() == [
        "Card",
        "Legitimate",
        "Identity",
        "Legitimate",
    ]


def test_extract_temporal_features(sample_data):
    df_result = extract_temporal_features(sample_data)
    assert "transaction_hour" in df_result.columns
    assert "transaction_day_of_week" in df_result.columns
    assert "is_weekend" in df_result.columns

    # 2023-07-06 is Thursday (3), 07-07 is Friday (4), 07-08 is Saturday (5), 07-09 is Sunday (6)
    assert df_result["transaction_hour"].tolist() == [21, 10, 8, 14]
    assert df_result["transaction_day_of_week"].tolist() == [3, 4, 5, 6]
    assert df_result["is_weekend"].tolist() == [0, 0, 1, 1]


def test_transform_numerical_distributions(sample_data):
    df_result = transform_numerical_distributions(sample_data)
    assert "amount_log" in df_result.columns
    # log1p of 0 is 0
    assert df_result["amount_log"][2] == 0.0
    assert df_result["amount_log"][0] > 0
