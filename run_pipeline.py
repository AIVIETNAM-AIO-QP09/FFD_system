"""Main pipeline executor for Financial Fraud Detection."""

from pathlib import Path

from src.data_loader import load_config, load_raw_data
from src.logger import get_logger
from src.preprocessing import run_baseline_pipeline

logger = get_logger(__name__)


def main():
    logger.info("=============================================================")
    logger.info("STARTING FINANCIAL FRAUD DETECTION PIPELINE RUNNER (WEEK 1)")
    logger.info("=============================================================")

    config_path = "configs/config.yaml"
    config = load_config(config_path)

    raw_path = config["paths"]["raw_data_path"]
    processed_path = config["paths"]["processed_data_path"]

    try:
        raw_df = load_raw_data(raw_path)
    except FileNotFoundError:
        logger.error(
            "Pipeline Execution Aborted. Verify source file location: %s", raw_path
        )
        return

    # Process data through modular functions
    cleaned_df = run_baseline_pipeline(raw_df)

    # Resolve output directory properly
    project_root = Path(__file__).resolve().parent
    full_processed_path = project_root / processed_path

    output_dir = full_processed_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directory successfully deployed or verified: %s", output_dir)

    logger.info("Saving final cleaned artifact to: %s", full_processed_path)
    cleaned_df.to_csv(full_processed_path, index=False)

    logger.info("=============================================================")
    logger.info(
        "SUCCESS: DATA PIPELINE CONCLUDED. HANDOVER SHAPE: %s", cleaned_df.shape
    )
    logger.info("=============================================================")


if __name__ == "__main__":
    main()
