# Streamlit Analyst Dashboard

This project includes an interactive Streamlit dashboard to demonstrate the **Fraud Analyst Co-pilot**.

## How to Run

1. Open your terminal in the project root.
2. Install the required dependency if you haven't already:
```bash
pip install streamlit
streamlit run app_streamlit.py
```

## LM Studio optional mode:

1. Open LM Studio.
2. Load Qwen3-8B GGUF Q4_K_M.
3. Start Local Server.
4. Confirm endpoint: `http://127.0.0.1:1234/v1`.
5. Run Streamlit dashboard.

**Note:** The dashboard works without LM Studio by using deterministic fallback reports.
