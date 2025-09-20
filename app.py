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
st.caption("Analyze text columns in a CSV to surface frequent phrases and near-duplicate lines.")

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


def tokenize(text: str, cjk_mode: str = "raw", jieba_mod=None):
    """
    cjk_mode:
      - "raw": keep each CJK run as one token (default)
      - "jieba": use jieba.lcut if available
      - "char-ngrams": return single CJK characters here; char n-grams are built later
    """
    toks = []
    for is_cjk, seg in split_runs(text):
        if not seg.strip():
            continue
        if is_cjk:
            if cjk_mode == "jieba" and jieba_mod is not None:
                toks.extend([w for w in jieba_mod.lcut(seg) if w.strip()])
            elif cjk_mode == "char-ngrams":
                toks.extend(list(seg))  # single characters; char n-grams added in miner
            else:
                toks.append(seg)  # raw: keep whole CJK segment
        else:
            toks.extend(TOKEN_RE.findall(seg))
    return toks


def _is_cjk_string(s: str) -> bool:
    return len(s) > 0 and all(bool(CJK_RE.match(ch)) for ch in s)


# ---------------------------
# Quality helpers: substring suppression and stitching
# ---------------------------
def suppress_substrings(cands):
    """
    Suppress shorter substrings when a longer/better phrase exists.
    cands: list[dict] with keys: phrase, doc_freq, total_freq, score(optional)
    """
    cands = sorted(cands, key=lambda x: (-(x.get("score") or 0), -x["doc_freq"], -len(x["phrase"])))
    keep = []
    seen = []
    for item in cands:
        p = item["phrase"]
        if any((p in q and p != q) for q in seen):
            continue
        keep.append(item)
        seen.append(p)
    return keep


def stitch_cjk_phrases(cands_by_n):
    """
    Stitch common 2/3/4-gram CJK phrases into longer ones by overlapping 1 char.
    Returns a list[str] of stitched candidates (reference only).
    """
    grams2 = {g for g, _, _ in cands_by_n.get(2, []) if _is_cjk_string(g)}
    grams3 = {g for g, _, _ in cands_by_n.get(3, []) if _is_cjk_string(g)}
    grams4 = {g for g, _, _ in cands_by_n.get(4, []) if _is_cjk_string(g)}
    stitched = set()

    def overlap_join(a, b):
        if len(a) >= 1 and len(b) >= 1 and a[-1] == b[0]:
            return a + b[1:]
        return None

    sources = [grams2, grams3, grams4]
    for _ in range(2):
        new_set = set()
        pool = set().union(*sources)
        for x in pool:
            for y in pool:
                if x == y:
                    continue
                j = overlap_join(x, y)
                if j and _is_cjk_string(j) and 2 <= len(j) <= 8:
                    new_set.add(j)
        stitched |= new_set
        sources.append(new_set)

    return sorted(stitched, key=len, reverse=True)


# ---------------------------
# Cached miners
# ---------------------------
@st.cache_data(show_spinner=False)
def mine_ngrams(rows,
                n_min=1,
                n_max=4,
                min_df=8,
                cjk_mode="raw",
                cjk_char_ng_min=2,
                cjk_char_ng_max=4,
                top_k=500,
                jieba_enabled=False):
    """
    Mine n-grams by document frequency and total frequency.

    Always adds CJK character-level n-grams (2..cjk_char_ng_max) from CJK runs,
    so that short terms like 自爆/自嘲 appear even in raw/jieba modes.

    Counters are initialized up to max_n_needed to avoid KeyError.
    """
    max_n_needed = max(n_max, cjk_char_ng_max)

    df_counters = {n: Counter() for n in range(n_min, max_n_needed + 1)}
    tf_counters = {n: Counter() for n in range(n_min, max_n_needed + 1)}

    jieba_mod = try_import_jieba() if (cjk_mode == "jieba" and jieba_enabled) else None

    for r in rows:
        toks = tokenize(r, cjk_mode=cjk_mode, jieba_mod=jieba_mod)

        # token-level grams
        for n in range(n_min, n_max + 1):
            if len(toks) >= n:
                grams = [' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)]
                grams = [g for g in grams if not any(is_noise(tok) for tok in g.split())]
                tf_counters[n].update(grams)
                for g in set(grams):
                    df_counters[n][g] += 1

        # character-level CJK grams (always on)
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

    # simple Dice score for n>=2 as a quality signal
    def dice_for_phrase(p, rows_norm):
        if _is_cjk_string(p):
            if len(p) < 2:
                return 0.0
            A = sum(1 for r in rows_norm if p[:-1] in r)
            B = sum(1 for r in rows_norm if p[1:] in r)
            AB = sum(1 for r in rows_norm if p in r)
        else:
            left = p.split(' ')[0]
            right = p.split(' ')[-1]
            A = sum(1 for r in rows_norm if (' ' + left + ' ') in (' ' + r + ' '))
            B = sum(1 for r in rows_norm if (' ' + right + ' ') in (' ' + r + ' '))
            AB = sum(1 for r in rows_norm if p in r)
        if A + B == 0:
            return 0.0
        return 2 * AB / (A + B)

    # assemble results with quality sorting and substring suppression
    results = {}
    rows_norm = rows
    for n in range(n_min, max_n_needed + 1):
        raw = [(g, df, tf_counters[n][g]) for g, df in df_counters[n].items() if df >= min_df]
        enriched = []
        for g, dfv, tfv in raw:
            score = dice_for_phrase(g, rows_norm) if n >= 2 else 0.0
            enriched.append({"phrase": g, "doc_freq": dfv, "total_freq": tfv, "score": score})
        enriched = suppress_substrings(enriched)
        enriched = sorted(enriched, key=lambda x: (-(x["score"] or 0), -x["doc_freq"], -len(x["phrase"])))[:top_k]
        results[n] = [(e["phrase"], e["doc_freq"], e["total_freq"]) for e in enriched]
    return results


@st.cache_data(show_spinner=False)
def find_near_duplicates(rows, threshold=92, prefix_len=24, cap=120):
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
# Sidebar controls (minimal)
# ---------------------------
st.sidebar.header("Settings")

# Max n-gram size: default 4, allowed 2..5
ngrams_max = st.sidebar.slider("Max n-gram size", 2, 5, 4, 1)

# Show top-K: default to max value (500)
top_k = st.sidebar.slider("Show top-K phrases per n", 50, 500, 500, 10)

# CJK tokenization mode: default raw
def _has_jieba():
    try:
        import jieba  # noqa: F401
        return True
    except Exception:
        return False

cjk_mode = st.sidebar.selectbox(
    "CJK tokenization mode",
    ["raw", "jieba", "char-ngrams"],
    index=0,
    help="raw keeps whole CJK runs; jieba uses Chinese word segmentation; char-ngrams splits into characters."
)
enable_jieba = (cjk_mode == "jieba" and _has_jieba())

# Single essential quality threshold
min_df = st.sidebar.slider("Min document frequency", 2, 50, 8, 1)

# Near-duplicate scanning: keep on by default, with fixed parameters
enable_dups = st.sidebar.checkbox("Scan near-duplicate rows", value=True)

# ---------------------------
# File upload (concise, no duplicate wording)
# ---------------------------
uploaded = st.file_uploader(
    "CSV file (UTF-8)",
    type=["csv"],
    help="Include text columns such as Title, Description, Caption, Text."
)

if not uploaded:
    # Keep empty state minimal; no extra repeated message
    st.stop()

df = pd.read_csv(uploaded)

# Choose text columns
text_cols_all = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("string")]
default_cols = [c for c in text_cols_all if c.lower() in ("title", "description", "caption", "text", "name")] or text_cols_all[:2]
col_text = st.multiselect(
    "Select text columns",
    options=text_cols_all,
    default=default_cols,
    help="Selected columns will be concatenated for analysis."
)
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

# N-gram mining (always add CJK char n-grams 2..ngrams_max for CJK segments)
st.subheader("Top n-grams (largest n first)")
with st.spinner("Mining n-grams..."):
    mined = mine_ngrams(
        rows,
        n_min=1,
        n_max=ngrams_max,
        min_df=min_df,
        cjk_mode=cjk_mode,
        cjk_char_ng_min=2,
        cjk_char_ng_max=max(ngrams_max, 6),  # allow longer CJK phrases up to 6 characters
        top_k=top_k,
        jieba_enabled=enable_jieba,
    )

# Display n in descending order
ordered_ns = list(range(1, max(ngrams_max, 1) + 1))[::-1]
cols = st.columns(min(3, len(ordered_ns)))
for idx, n in enumerate(ordered_ns):
    df_show = pd.DataFrame(mined.get(n, []), columns=["ngram", "doc_freq", "total_freq"])

    # hide single-character CJK unigram by default
    if n == 1 and not df_show.empty:
        mask = ~df_show["ngram"].map(lambda x: len(x) == 1 and _is_cjk_string(x))
        df_show = df_show[mask]

    with cols[idx % len(cols)]:
        st.markdown(f"{n}-grams")
        st.dataframe(df_show, use_container_width=True)

# Stitched CJK phrases as reference
st.markdown("CJK stitched phrases (reference)")
stitched = stitch_cjk_phrases({n: mined.get(n, []) for n in (2, 3, 4)})
df_stitched = pd.DataFrame({"phrase": stitched[:50]})
st.dataframe(df_stitched, use_container_width=True)

# Near-duplicate clusters (optional)
st.subheader("Near-duplicate rows")
if enable_dups:
    with st.spinner("Scanning for near-duplicates..."):
        pairs = find_near_duplicates(rows, threshold=92, prefix_len=24, cap=120)
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
else:
    st.info("Duplicate scan is turned off.")

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
st.download_button("normalized_texts.csv", out_norm.getvalue(), "normalized_texts.csv", "text/csv")

# Annotated n-grams (flattened)
annot_rows = []
for n, grams in mined.items():
    for g, dfreq, tfreq in grams:
        annot_rows.append({"n": n, "ngram": g, "doc_freq": dfreq, "total_freq": tfreq})
out_ngrams = io.StringIO()
pd.DataFrame(annot_rows).to_csv(out_ngrams, index=False)
st.download_button("ngrams.csv", out_ngrams.getvalue(), "ngrams.csv", "text/csv")
