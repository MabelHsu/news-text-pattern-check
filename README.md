# News Text Pattern Check

A Streamlit app to **discover repetitive phrases** in FB/IG CSV exports.

## Features
- Upload CSV (titles, descriptions, captions)
- Auto-extract frequent **n-grams (1–5 words)**
- Highlight **near-duplicate rows**
- Export normalized text for further review

## Usage
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
