# Stoncks NQ! Alert GUI

This project provides a starter Python GUI application that trains a market-prediction model on historical NQ! data, then runs continuously to generate real-time alerts. The architecture is designed for extension to live trading (market, limit, SL/TP) and broker connectivity later.

> **Disclaimer:** This project is a starter template for research and education. It is not financial advice and does not guarantee profits.

## Features
- GUI dashboard using `tkinter`.
- Data pipeline with RSI, support/resistance, volume, and momentum features.
- Optional features such as FVG/iFVG and breakout flags (only used if present in data).
- Model training and inference with a scikit-learn classifier.
- Commission-aware decision layer (assumes $1 per contract).
- Streaming interface stub that can later be wired to WebSocket or broker APIs.

## Quick start
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place historical NQ! data into `data/nq.csv`. The CSV should include at least:
   `timestamp,open,high,low,close,volume`.
3. Train the model:
   ```bash
   python -m stoncks_app.cli train --data data/nq.csv --model artifacts/model.pkl
   ```
4. Launch the GUI:
   ```bash
   python -m stoncks_app.cli run --model artifacts/model.pkl --data data/nq.csv
   ```

## Project layout
```
stoncks_app/
  app.py        # tkinter GUI and controls
  cli.py        # CLI entrypoint
  config.py     # runtime settings
  data.py       # feature engineering and dataset prep
  model.py      # training + inference wrappers
  stream.py     # streaming data stub for realtime integration
  trading.py    # signal logic with commission-aware profitability checks
```

## Next steps
- Replace `stream.py` with a real WebSocket feed (broker or data vendor).
- Extend `trading.py` to push orders to a broker API.
- Add robust backtesting and risk management before live deployment.
