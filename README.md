# Campaign AI — Streamlit App

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Streamlit:

```bash
streamlit run app.py
```

The app expects `products_campaign_sales.csv` in the same folder.

## App sections

- Dashboard
- Predict Orders
- Explore Data
- Model Insights
- Methodology

The app trains the documented tuned Extra Trees model automatically when it starts, so no separate model file is required.
