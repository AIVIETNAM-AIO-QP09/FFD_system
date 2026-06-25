# 🛡️ Financial Fraud Detection System - Core Pipeline

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Build: Passing](https://img.shields.io/badge/build-passing-brightgreen)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20AI-5C5CFF)

Welcome to the **Financial Fraud Detection** repository. This project is designed to process massive volumes of financial transaction logs (5M+ records) to identify fraudulent patterns. 

Currently, this repository houses the **Week 1 Baseline Data Pipeline**, which focuses strictly on robust data ingestion, structural cleaning, and baseline temporal feature extraction without data leakage.

---

## 🏗️ Repository Architecture

Our structure strictly adheres to the **Cookiecutter Data Science** enterprise standard, ensuring separation of concerns and easy onboarding for all Dev Team members.

```text
FFD_system/
├── configs/             # YAML configurations (No hardcoded variables in code!)
├── data/                # Data storage & demo transaction subset artifacts
├── docs/                # Extended project documentation and team onboarding guides
├── models/              # Serialized LightGBM models, preprocessors & RAG retrievers
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── reports/             # Archived benchmark analysis and model audit reports
├── scripts/             # Categorized execution runners
│   ├── audits/          # Failure analysis, sanity check & metric audits
│   ├── eda/             # Exploratory analysis & feature discovery scripts
│   └── modeling/        # Baseline training, model comparison & tuning runners
├── src/                 # Core reusable Python library modules
│   ├── logger.py        # Centralized logging configuration
│   ├── data_loader.py   # Secure data ingestion and YAML parsing
│   ├── preprocessing.py # In-place memory-optimized cleaning functions
│   ├── inference.py     # Batch and real-time prediction pipelines
│   ├── llm_client.py    # Local Qwen3 AI copilot client
│   └── dashboard_utils.py # UI helper utilities
├── tests/               # Pytest unit tests verifying preprocessing & copilot logic
├── app_streamlit.py     # Interactive Fraud Analyst Streamlit UI
├── run_pipeline.py      # Entry point executor for the data pipeline
├── Dockerfile & docker-compose.yml # Containerization configurations
├── requirements.txt     # Locked Python dependencies
├── Makefile             # One-click automation commands for Devs
├── pyproject.toml       # Black & Isort formatter configurations
└── .env                 # Local IDE environment variables (PYTHONPATH)
```

---

## 🚀 Getting Started (Onboarding)

### ⚡ Instant Onboarding with Docker Compose (Recommended)
For new team members or quick demonstrations, you do **not** need to install local Python dependencies. Simply ensure Docker Desktop is running and execute:

```bash
docker-compose up -d --build
```
Access the **Fraud Analyst Dashboard** immediately at `http://localhost:8501`.

---

### 1. Manual Prerequisites
Ensure you have **Python 3.12 or higher** installed. It is highly recommended to use a Virtual Environment to avoid global package conflicts.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Setup & Installation
Instead of running complex commands, use the provided `Makefile` to install all dependencies and development tools (`pytest`, `black`, `flake8`).

```bash
make setup
```

### 3. Data Provisioning
Due to data privacy and size constraints, raw data is **not** included in version control. 
- Obtain the `financial_fraud_detection_dataset.csv` file from the Data Engineering team.
- Place it directly into the `data/raw/` directory.

---

## ⚙️ Operating the Pipeline

**To execute the end-to-end preprocessing pipeline:**
```bash
make run
```

### What does the pipeline actually do?
When you trigger the pipeline, the system sequentially performs:
1. **Logical Anomaly Correction**: Fixes system-generated glitches (e.g., negative `time_since_last_transaction`).
2. **Conditional Imputation**: Intelligently fills missing metadata (e.g., `fraud_type` for legitimate users).
3. **Temporal Extraction**: Parses ISO8601 timestamps into ML-ready cyclical features (`hour`, `day_of_week`, `is_weekend`).
4. **Numerical Transformation**: Applies `log1p` normalization to heavily right-skewed financial metrics (like transaction `amount`).

The final artifact is dumped to `data/processed/cleaned_baseline_data.csv`.

---

## 🛠️ Development & Maintenance Guidelines

To maintain our enterprise codebase quality, please adhere to the following when extending the repo:

### 1. Code Formatting
We use `Black` and `Isort` to guarantee a uniform code style across the team. **Never format code manually.**
```bash
make format
```

### 2. Quality Assurance (Linting)
Before submitting a Pull Request, ensure your code passes our `flake8` linter rules (Max line length: 88).
```bash
make lint
```

### 3. Unit Testing
Every new preprocessing function must have a corresponding test in `tests/test_preprocessing.py`. We use dummy data fixtures to validate logic.
```bash
make test
```

### 4. Memory Management Rules
When working with 5M+ rows, **do not use `df.copy()`** inside individual functions. Mutate the Pandas DataFrame in-place (`df["col"] = ...`) to prevent Out-Of-Memory (OOM) server crashes.

---

## 💡 Troubleshooting

- **VSCode says "Import 'src' could not be resolved":** 
  This happens because the IDE doesn't know where the project root is. We have provided a `.env` file containing `PYTHONPATH=.`. Simply reload your VSCode window (`Ctrl+Shift+P` -> `Developer: Reload Window`), and Pylance will read it correctly.
- **Pipeline fails with Memory Error:** 
  Ensure you are not running other heavy applications. If scaling to larger datasets, consider transitioning the `pandas` logic to `polars` or `PySpark` in future iterations.

---

## 🌍 Version Control & Git Workflow

The central repository is hosted at: **[AIVIETNAM-AIO-QP09/FFD_system](https://github.com/AIVIETNAM-AIO-QP09/FFD_system)**

To ensure a smooth collaboration process, please follow these Git practices:
1. **Always pull before coding:** `git pull origin main`
2. **Never push directly to main:** Create a new branch for your feature (`git checkout -b feature/your-feature-name`).
3. **Commit often with descriptive messages:** Ensure your code passes `make format` and `make test` before committing.
4. **Push and Create a Pull Request:** `git push origin feature/your-feature-name` and request a review from a team member.

---

## Local LLM Co-pilot with LM Studio

1. Install LM Studio.
2. Download Qwen3 8B GGUF Q4_K_M.
3. Load the model.
4. Start Local Server.
5. Confirm server is reachable at http://127.0.0.1:1234.
6. Run:

```bash
python run_lmstudio_copilot_demo.py
```

### Limitations:
* The underlying model is a weak fraud-ranking signal.
* RAG retrieves similar structured transactions, not legal proof.
* The LLM improves report generation and explainability, not classification accuracy.
* Human review remains required.
* Do not use this system for automatic blocking or fund freezing.
