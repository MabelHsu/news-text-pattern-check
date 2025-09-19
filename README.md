# News Text Pattern Self-Check

A Streamlit app to discover repetitive phrases and template-like structures in CSV exports (e.g., Facebook/Instagram titles, descriptions, captions). It supports mixed Chinese/English text and can surface attribution-heavy phrasing, platform terms, and near-duplicate headlines.

## Features
- Upload CSV and select which text columns to analyze
- Language-aware normalization (lowercase Latin, retain CJK)
- N-gram mining (1–5) with document frequency
- Chinese character-level n-grams to capture short words like 「自爆」「自嘲」
- Near-duplicate headline/description detection using fuzzy matching
- Keyword quick check
- Download normalized and annotated results

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
