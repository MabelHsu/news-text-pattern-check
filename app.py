import re
import io
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from rapidfuzz import fuzz

# ---------------------------
# Helpers
# ---------------------------
CJK_RE = re.compile(r'[\u4e00-\u9fff]')
TOKEN_RE = re.compile(r'[\u4e00-\u9fffA-Za-z0-9_]+')


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = ''.join([ch.lower() if not CJK_RE.match(ch) else ch for ch in s])
    return re.sub(r'\s+', ' ', s.strip())


def tokenize(text: str):
    return TOKEN_RE.findall(text)


STOP_TOKENS = set("""
的 了 在 是 和 與 地 得 也 及 並 或 你 我 他 她 它 我們 你們 他們 這 那 the a an and or of to in for on with is are was were be been at by from
""".split())


def is_noise(ng):
    if ng.isdigit(): return True
    if ng in STOP_TOKENS: return True
    if len(ng) == 1 and not CJK_RE.match(ng): return True
    return False


def mine_ngrams(rows, n_min=1, n_max=5, min_df=5):
    df_counters = {n:Counter() for n in range(n_min, n_max+1)}
    tf_counters = {n:Counter() for n in range(n_min, n_max+1)}
    for r in rows:
        toks = [t for t in tokenize(r)]
        for n in range(n_min, n_max+1):
            if len(toks) < n: continue
            seen = set()
            grams = [' '.join(toks[i:i+n]) for i in range(len(toks)-n+1)]
            tf_counters[n].update(grams)
            for g in set(grams):
                seen.add(g)
            for g in seen:
                df_counters[n][g] += 1
    results = {}
    for n in range(n_min, n_max+1):
        cand = []
        for g, df in df_counters[n].items():
            if df >= min_df:
                if any(is_noise(tok) for tok in g.split()): 
                    continue
                cand.append((g, df, tf_counters[n][g]))
        results[n] = sorted(cand, key=lambda x:(-x[1], -x[2]))
    return results


def find_near_duplicates(rows, threshold=90, prefix_len=20, cap=120):
    buckets = defaultdict(list)
    for i, t in enumerate(rows):
        buckets[t[:prefix_len]].append((i, t))
    pairs = []
    for items in buckets.values():
        if len(items) > cap: items = items[:cap]
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                ii, a = items[i]
                jj, b = items[j]
                s = fuzz.token_set_ratio(a, b)
                if s >= threshold:
                    pairs.append((ii, jj, s))
    return pairs

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="News Pattern Check", layout="wide")
st.title("News Pattern Self-Check Tool")
st.caption("Upload a CSV (FB/IG exports) → discover repetitive patterns and possible originality risks.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    text_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    if not text_cols:
        st.error("No text columns found")
        st.stop()

    col_text = st.multiselect("Select text columns", text_cols, default=text_cols[:2])
    df["_TEXT_RAW"] = df[col_text].astype(str).fillna("").agg(" ".join, axis=1)
    rows = [norm_text(x) for x in df["_TEXT_RAW"].tolist()]

    st.subheader("Top n-grams (frequent phrases)")
    mined = mine_ngrams(rows, n_min=1, n_max=3, min_df=5)
    for n, grams in mined.items():
        st.markdown(f"**{n}-grams**")
        st.dataframe(pd.DataFrame(grams[:30], columns=["ngram", "doc_freq", "total_freq"]))

    st.subheader("Near-duplicate rows")
    pairs = find_near_duplicates(rows, threshold=90)
    st.write(f"Found {len(pairs)} near-duplicate pairs")
    if pairs:
        sample = []
        for (i,j,s) in pairs[:100]:
            sample.append({"i": i, "j": j, "similarity": s,
                           "text_i": df.loc[i, "_TEXT_RAW"], "text_j": df.loc[j, "_TEXT_RAW"]})
        st.dataframe(pd.DataFrame(sample))

    st.subheader("⬇Download annotated report")
    out = io.StringIO()
    pd.DataFrame(rows, columns=["normalized"]).to_csv(out, index=False)
    st.download_button("Download normalized_texts.csv", out.getvalue(), "normalized_texts.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Tips:** High-frequency phrases and near-duplicate structures often explain why content looks repetitive to the monetisation system.")
