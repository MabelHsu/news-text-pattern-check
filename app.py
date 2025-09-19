import io
import re
from collections import Counter, defaultdict

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="News Text Pattern Self-Check", layout="wide")
st.title("News Text Pattern Self-Check")
st.caption("Upload a CSV and discover repetitive phrases and near-duplicate headlines/descriptions.")

# ---------------------------
# Language-aware helpers
# ---------------------------
CJK_RE = re.compile(r'[\u4e00-\u9fff]')
TOKEN_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9_]+')


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = ''.join([ch.lower() if not CJK_RE.match(ch) else ch for ch in s])
    s = re.sub(r'\s+', ' ', s.strip())
    return s


def split_runs(text: str):
    """Split into [(is_cjk, segment), ...] so CJK and non-CJK can be handled differently."""
    runs = []
    if not text:
        return runs
    cur_is_cjk = bool(CJK_RE.match(text[0]))
    buf = []
    for ch in text:
        is
