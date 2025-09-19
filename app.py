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
        is_cjk = bool(CJK_RE.match(ch))
        if is_cjk == cur_is_cjk:
            buf.append(ch)
        else:
            runs.append((cur_is_cjk, ''.join(buf)))
            buf = [ch]
            cur_is_cjk = is_cjk
    if buf:
        runs.append((cur_is_cjk, ''.join(buf)))
    return runs


def cjk_char_ngrams(seg: str, n_min=2, n_max=4):
    grams = []
    L = len(seg)
    for n in range(n_min, n_max + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            grams.append(seg[i:i + n])
    return grams


STOP_TOKENS = set("""
的 了 在 是 和 與 地 得 也 及 並 或 你 我 他 她 它 我們 你們 他們 這 那 the a an and or of to in for on with is are was were be been at by from
""".split())


def is_noise(ng: str) -> bool:
    if ng.isdigit():
        return True
    if ng in STOP_TOKENS:
        return True
    if len(ng) == 1 and not CJK_RE.match(ng):
        return True
    return False


def try_import_jieba():
    try:
        import jieba  # type: ignore
        return jieba
    except Exception:
        return None


def tokenize(text: str, cjk_mode: str = "char-ngrams", jieba_mod=None):
    """
    cjk_mode:
      - "jieba": use jieba.lcut if available
      - "char-ngrams": return single CJK characters here; n-grams built later
      - "raw": keep each CJK run as one token (not recommended)
    """
    toks = []
    for is_cjk, seg in split_runs(text):
        if not seg.strip():
            continue
        if is_cjk:
            if cjk_mode == "jieba" and jieba_mod is not None:
                toks.extend([w for w in jieba_mod.lcut(seg) if w.strip()])
            elif cjk_mode == "char-ngrams":
                toks.extend(list(seg))  # single characters; n-grams added in miner
            else:
                toks.append(seg)  # raw
        else:
            toks.extend(TOKEN_RE.findall(seg))
    return toks


# ---------------------------
# Cached miners
# ---------------------------
@st.cache_data(show_spinner=False)
def mine_ngrams(rows,
                n_min=1,
                n_max=3,
                min_df=5,
                cjk_mode="char-ngrams",
                cjk_char_ng_min=2,
                cjk_char_ng_max=4,
                top_k=100,
                jieba_enabled=False):
    """
    Mine n-grams by document frequency and total frequency.

    Counters are initialized up to max_n = max(n_max, cjk_char_ng_max)
    to avoid KeyError when CJK n-gram max exceeds token-level n_max.
    """
    max_n = max(n_max, cjk_char_ng_max)
    df_counters = {n: Counter() for n in range(n_min, max_n + 1)}
    tf_counters = {n: Counter() for n in range(n_min, max_n + 1)}

    jieba_mod = try_import_jieba() if (cjk_mode == "jieba" and jieba_enabled) else None

    for r in rows:
        toks = tokenize(r, cjk_mode=cjk_mode, jieba_mod=jieba_mod)

        # token-level grams (non-CJK and whatever tokenize returns)
        for n in range(n_min, n_max + 1):
            if len(toks) >= n:
                grams = [' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
                grams = [g for g in grams if not any(is_noise(tok) for tok in g.split())]
                tf_counters[n].update(grams)
                for g in set(grams):
                    df_counters[n][g] += 1

        # character-level CJK grams
        for is_cjk, seg in split_runs(r):
            if not is_cjk:
                continue
            for n in range(max(2, cjk_char_ng_min), max(2, cjk_char_ng_max) + 1):
                grams = cjk_char_ngrams(seg, n_min=n, n_max=n)
                if not grams:
                    continue
                tf_counters[n].update(grams)
                for g in set(grams):
                    df_counters[n][g] += 1

    # collect results
    results = {}
    for n in range(n_min, max_n + 1):
        cand = [(g, df, tf_counters[n][g]) for g, df in df_counters[n].items() if df >= min_df]
        cand = sorted(cand, key=lambda x: (-x[1], -x[2]))[:top_k]
        results[n] = cand
    return results


@st.cache_data(show_spinner=False)
def find_near_duplicates(rows, threshold=90, prefix_len=20, cap=120):
    """Lightweight fuzzy duplicate detection with blocking."""
    buckets = defaultdict(list)
    for i, t in enumerate(rows):
        buckets[t[:prefix_len]].append((i, t))
    pairs = []
    for items in buckets.values():
        if len(items) > cap:
            items = items[:cap]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                ii, a = items[i]
                jj, b = items[j]
                s = fuzz.token_set_ratio(a, b)
                if s >= threshold:
                    pairs.append((ii, jj, s))
    return pairs


# ---------------------------
# Sidebar controls (friendly defaults)
# ---------------------------
st.sidebar.header("Settings")

def _has_jieba():
    try:
        import jieba  # noqa: F401
        return True
    except Exception:
        return False

ngrams_max = st.sidebar.selectbox("Max n-gram size", [1, 2, 3, 4, 5], index=2)  # default 3
min_df = st.sidebar.slider("Min document frequency (phrase must appear in at least this many rows)", 2, 50, 12, 1)
top_k = st.sidebar.slider("Show top-K phrases per n", 20, 300, 60, 10)

_default_cjk_index = 1 if _has_jieba() else 0
cjk_mode = st.sidebar.selectbox("CJK tokenization mode", ["char-ngrams", "jieba", "raw"], index=_default_cjk_index)

cjk_char_ng_min = st.sidebar.slider("CJK character n-gram min", 2, 4, 2, 1)
cjk_char_ng_max = st.sidebar.slider("CJK character n-gram max", 2, 6, 3, 1)
enable_jieba = st.sidebar.checkbox("Enable jieba (requires package installed)", value=(cjk_mode == "jieba" and _has_jieba()))

dup_threshold = st.sidebar.slider("Near-duplicate similarity threshold (token_set_ratio ≥)", 70, 100, 92, 1)
prefix_len = st.sidebar.slider("Duplicate blocking prefix length", 10, 60, 24, 2)
pairs_cap = st.sidebar.slider("Max comparisons per bucket (cap)", 50, 400, 100, 10)

hide_single_cjk_unigram = st.sidebar.checkbox("Hide single-character CJK unigrams in tables", value=True)

# ---------------------------
# File upload
# ---------------------------
uploaded = st.file_uploader("Upload CSV (UTF-8). Include columns like Title, Description, Caption, Text.", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

# Choose text columns
text_cols_all = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
default_cols = [c for c in text_cols_all if c.lower() in ("title", "description", "caption", "text", "name")] or text_cols_all[:2]
col_text = st.multiselect("Select text columns to analyze", options=text_cols_all, default=default_cols, help="Selected columns will be concatenated for analysis.")
if not col_text:
    st.warning("Please select at least one text column.")
    st.stop()

# Build merged text
df["_TEXT_RAW"] = df[col_text].astype(str).fillna("").agg(" ".join, axis=1)
rows = [norm_text(x) for x in df["_TEXT_RAW"].tolist()]

# KPIs
k1, k2 = st.columns(2)
k1.metric("Rows analyzed", f"{len(rows)}")
k2.metric("Selected columns", f"{', '.join(col_text)}")

# N-gram mining
st.subheader("Top n-grams")
with st.spinner("Mining n-grams..."):
    mined = mine_ngrams(
        rows,
        n_min=1,
        n_max=ngrams_max,
        min_df=min_df,
        cjk_mode=cjk_mode,
        cjk_char_ng_min=cjk_char_ng_min,
        cjk_char_ng_max=cjk_char_ng_max,
        top_k=top_k,
        jieba_enabled=enable_jieba,
    )

def _is_cjk_string(s: str) -> bool:
    return all(bool(CJK_RE.match(ch)) for ch in s)

cols = st.columns(min(3, ngrams_max))
for idx, n in enumerate(range(1, ngrams_max + 1)):
    df_show = pd.DataFrame(mined.get(n, []), columns=["ngram", "doc_freq", "total_freq"])

    # Hide single-character CJK unigrams if checked
    if n == 1 and hide_single_cjk_unigram and not df_show.empty:
        mask = ~df_show["ngram"].map(lambda x: len(x) == 1 and _is_cjk_string(x))
        df_show = df_show[mask]

    with cols[idx % len(cols)]:
        st.markdown(f"{n}-grams")
        st.dataframe(df_show, use_container_width=True)

# Near-duplicate clusters
st.subheader("Near-duplicate rows")
with st.spinner("Scanning for near-duplicates..."):
    pairs = find_near_duplicates(rows, threshold=dup_threshold, prefix_len=prefix_len, cap=pairs_cap)
st.write(f"Pairs found: {len(pairs)}")
if pairs:
    sample = []
    for (i, j, s) in pairs[:200]:
        sample.append({
            "row_i": i, "row_j": j, "similarity": s,
            "text_i": df.loc[i, "_TEXT_RAW"],
            "text_j": df.loc[j, "_TEXT_RAW"],
        })
    st.dataframe(pd.DataFrame(sample), use_container_width=True)

# Keyword quick check
st.subheader("Keyword quick check")
kw = st.text_input("Enter keyword (e.g., 自爆, 自嘲)")
if kw:
    hits = df[df["_TEXT_RAW"].astype(str).str.contains(kw, na=False)]
    st.write(f"Rows containing '{kw}': {len(hits)}")
    st.dataframe(hits[col_text + ["_TEXT_RAW"]].head(300), use_container_width=True)

# Downloads
st.subheader("Downloads")

# Normalized texts
out_norm = io.StringIO()
pd.DataFrame({"normalized_text": rows}).to_csv(out_norm, index=False)
st.download_button("Download normalized_texts.csv", out_norm.getvalue(), "normalized_texts.csv", "text/csv")

# Annotated n-grams (flattened)
annot_rows = []
for n, grams in mined.items():
    for g, dfreq, tfreq in grams:
        annot_rows.append({"n": n, "ngram": g, "doc_freq": dfreq, "total_freq": tfreq})
out_ngrams = io.StringIO()
pd.DataFrame(annot_rows).to_csv(out_ngrams, index=False)
st.download_button("Download ngrams.csv", out_ngrams.getvalue(), "ngrams.csv", "text/csv")
