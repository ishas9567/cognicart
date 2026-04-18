"""
CogniCart – Week 9: Streamlit Dashboard
Run:  streamlit run week9_cognicart_dashboard.py
Place this file at:  cognicart/notebooks/week9_cognicart_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CogniCart",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark teal theme */
:root {
    --primary: #0f766e;
    --accent:  #2dd4bf;
    --bg:      #0f172a;
    --card:    #1e293b;
    --border:  #334155;
    --text:    #ffffff;
    --muted:   #ffffff;
}

.stApp { background-color: var(--bg); color: var(--text); }


/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid var(--border);
    color: #ffffff !important;
}
/* Fix KPI metric text */
/* Fix KPI metric text */
[data-testid="metric-container"] label,
[data-testid="metric-container"] div,
[data-testid="metric-container"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #ffffff !important;
}

/* Fix sidebar text */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Cards */
.cogni-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.cogni-card h4 { color: var(--accent); margin: 0 0 8px 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
.cogni-card .value { font-size: 2rem; font-weight: 700; color: var(--text); }
.cogni-card .sub   { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }

/* Rule pills */
.rule-pill {
    display: inline-block;
    background: #134e4a;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px;
}

/* Segment badge */
.seg-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.seg-Premium  { background:#1d4ed8; color:#bfdbfe; }
.seg-Regular  { background:#065f46; color:#a7f3d0; }
.seg-Budget   { background:#92400e; color:#fde68a; }
.seg-Inactive { background:#4b5563; color:#d1d5db; }

/* Header */
.cogni-header {
    background: linear-gradient(135deg, #0f766e 0%, #0f172a 60%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    border: 1px solid #0f766e;
}
.cogni-header h1 { color: #2dd4bf; font-size: 2.2rem; font-weight: 700; margin: 0; }
.cogni-header p  { color: #94a3b8; margin: 6px 0 0 0; font-size: 0.95rem; }

/* Chat bubbles */
.chat-user     { background:#1e3a5f; border-radius:12px 12px 4px 12px; padding:12px 16px; margin:8px 0; color:#e2e8f0; }
.chat-cogni    { background:#134e4a; border-radius:12px 12px 12px 4px; padding:12px 16px; margin:8px 0; color:#e2e8f0; border-left:3px solid #2dd4bf; }
.chat-label    { font-size:0.7rem; color:#64748b; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px; }

/* Metrics override */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
}

/* Button */
.stButton > button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
}
.stButton > button:hover { background: #0d9488; }

/* Selectbox / text input */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached so they run once)
# ══════════════════════════════════════════════════════════════════════════════

BASE = "cognicart"   # adjust if running from a different working directory

@st.cache_data
def load_transactions():
    path = f"{BASE}/data/cleaned/supermarket_clean.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_rules():
    for fname in ["enhanced_rules.csv", "association_rules.csv"]:
        path = f"{BASE}/data/cleaned/{fname}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df
    return None

@st.cache_data
def load_rfm():
    for fname in ["rfm_with_segments.csv", "rfm_table.csv"]:
        path = f"{BASE}/data/cleaned/{fname}"
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_data
def load_similarity():
    path = f"{BASE}/data/cleaned/product_similarity.csv"
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_clusters():
    path = f"{BASE}/data/cleaned/product_clusters.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def clean_frozenset(s):
    """Convert frozenset string like frozenset({'Milk'}) → 'Milk' """
    s = str(s)
    for tok in ["frozenset(", ")", "{", "}", "'"]:
        s = s.replace(tok, "")
    return s.strip()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def rule_based_rag(question, rules_df, rfm_df):
    q = question.lower()

    if rules_df is not None:
        rules_df["antecedents_clean"] = rules_df["antecedents"].apply(clean_frozenset)
        rules_df["consequents_clean"] = rules_df["consequents"].apply(clean_frozenset)
        products_in_q = [
            p for p in rules_df["antecedents_clean"].unique()
            if p.lower() in q
        ]
        if products_in_q:
            matches = rules_df[
                rules_df["antecedents_clean"].isin(products_in_q)
            ].sort_values("lift", ascending=False).head(5)
            if not matches.empty:
                recs = matches["consequents_clean"].tolist()
                recs_str = ", ".join(recs[:3])
                prod_str = " + ".join(products_in_q)
                return (
                    f"Based on {len(matches)} association rules, "
                    f"customers who buy {prod_str} also purchase: "
                    f"{recs_str}. "
                    f"Top rule lift: {matches.iloc[0]['lift']:.2f}, "
                    f"confidence: {matches.iloc[0]['confidence']:.0%}."
                )

    if rfm_df is not None and "Segment" in rfm_df.columns:
        for seg in ["Premium", "Budget", "Regular", "Inactive"]:
            if seg.lower() in q:
                sub = rfm_df[rfm_df["Segment"] == seg]
                spend_col = "Monetary" if "Monetary" in sub.columns else "Monetory"
                avg_spend = sub[spend_col].mean() if spend_col in sub.columns else 0
                freq = sub["Frequency"].mean() if "Frequency" in sub.columns else 0
                tips = {
                    "Premium":  "High-value customers — ideal for premium bundle offers.",
                    "Budget":   "Price-sensitive — focus on value packs and discounts.",
                    "Regular":  "Moderate buyers — target with loyalty rewards.",
                    "Inactive": "Low activity — consider re-engagement campaigns.",
                }
                return (
                    f"The {seg} segment has {len(sub)} customers. "
                    f"Average spend: Rs.{avg_spend:,.0f}. "
                    f"Average frequency: {freq:.1f} transactions. "
                    f"{tips.get(seg, '')}"
                )

    return (
        "Try mentioning a product name (e.g. Milk, Rice) "
        "or a segment name (Premium, Budget, Regular) for a specific answer."
    )
with st.sidebar:
    st.markdown("### 🛒 CogniCart")
    st.markdown("<span style='color:#94a3b8;font-size:0.8rem'>AI/ML Capstone Dashboard</span>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "🔗 Association Rules",
         "👥 Customer Segments",
         "🤖 BERT Similarity",
         "💬 Ask CogniCart (RAG)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.75rem;line-height:1.6'>
    <b style='color:#94a3b8'>Stack</b><br>
    FP-Growth · K-Means · LSTM<br>
    Sentence-BERT · LangChain<br>
    ChromaDB · RAG
    </div>
    """, unsafe_allow_html=True)


# Load all data once
df       = load_transactions()
rules_df = load_rules()
rfm_df   = load_rfm()
sim_df   = load_similarity()
clusters = load_clusters()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("""
    <div class='cogni-header'>
        <h1>🛒 CogniCart</h1>
        <p>A Cognitive AI/ML/DL Framework for Intelligent Market Basket Analysis using NLP and Large Language Models</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ Dataset not found. Make sure `cognicart/data/cleaned/supermarket_clean.csv` exists.")
        st.stop()

    # ── KPIs ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{df['TransactionID'].nunique():,}")
    with col2:
        st.metric("Unique Customers", f"{df['Customer_ID'].nunique():,}")
    with col3:
        st.metric("Products", f"{df['Product'].nunique()}")
    with col4:
        total_rev = (df["Quantity"] * df["Price"]).sum() if "Quantity" in df.columns and "Price" in df.columns else df["TotalAmount"].sum() if "TotalAmount" in df.columns else 0
        st.metric("Total Revenue", f"₹{total_rev:,.0f}")

    st.markdown("---")

    # ── Top products + Category split ───────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🏆 Top 10 Products")
        top_prods = df["Product"].value_counts().head(10).reset_index()
        top_prods.columns = ["Product", "Count"]
        st.bar_chart(top_prods.set_index("Product"))

    with col_b:
        st.subheader("📦 Sales by Category")
        if "Category" in df.columns:
            cat_sales = df.groupby("Category")["TotalAmount"].sum().sort_values(ascending=False) if "TotalAmount" in df.columns else df["Category"].value_counts()
            st.bar_chart(cat_sales)
        else:
            st.info("Category column not found in dataset.")

    # ── Monthly trend ───────────────────────────────────────────────────────
    st.subheader("📈 Monthly Transaction Trend")
    monthly = df.groupby(df["Date"].dt.to_period("M"))["TransactionID"].nunique()
    monthly.index = monthly.index.astype(str)
    st.line_chart(monthly)

    # ── Pipeline overview ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗺️ Project Pipeline")
    stages = [
        ("Week 1", "Dataset Creation", "500 customers · 8 categories · synthetic Indian supermarket"),
        ("Week 2", "Data Cleaning & EDA", "Missing values · duplicates · feature engineering · visualisations"),
        ("Week 3", "FP-Growth MBA",       "Association rules · lift · confidence · network graphs"),
        ("Week 4", "K-Means Clustering",  "RFM table · Elbow/Silhouette · 4 customer segments"),
        ("Week 5", "LSTM Deep Learning",  "Purchase sequence modelling · next-product prediction"),
        ("Week 6", "Sentence-BERT NLP",   "384-dim embeddings · cosine similarity · semantic clusters"),
        ("Week 7", "LLM Integration",     "Rule explanation · query answering · recommendation engine"),
        ("Week 8", "RAG Pipeline",        "ChromaDB vector store · LangChain · faithfulness evaluation"),
        ("Week 9", "Streamlit Dashboard", "← You are here"),
    ]
    for wk, title, desc in stages:
        done = wk != "Week 9"
        icon = "✅" if done else "🔵"
        st.markdown(f"{icon} **{title}** &nbsp;&nbsp; <span style='color:#64748b;font-size:0.85rem'>{desc}</span>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔗 Association Rules":
    st.title("🔗 Association Rules Explorer")

    if rules_df is None:
        st.error("Rules file not found. Run Week 3 notebook first.")
        st.stop()

    # Clean antecedent / consequent display
    rules_df["antecedents_clean"] = rules_df["antecedents"].apply(clean_frozenset)
    rules_df["consequents_clean"] = rules_df["consequents"].apply(clean_frozenset)

    # ── Filters ─────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.3, 0.05)
    with col2:
        min_lift = st.slider("Min Lift", 1.0, 10.0, 1.0, 0.5)
    with col3:
        top_n = st.selectbox("Show top N rules", [10, 20, 50, 100], index=0)

    filtered = rules_df[
        (rules_df["confidence"] >= min_conf) &
        (rules_df["lift"]       >= min_lift)
    ].sort_values("lift", ascending=False).head(top_n)

    st.markdown(f"<span style='color:#94a3b8'>Showing **{len(filtered)}** rules</span>", unsafe_allow_html=True)

    # Display table
    display_cols = ["antecedents_clean", "consequents_clean", "support", "confidence", "lift"]
    rename_map   = {"antecedents_clean": "IF (Antecedent)", "consequents_clean": "THEN (Consequent)"}
    if "SemanticSimilarity" in filtered.columns:
        display_cols.append("SemanticSimilarity")
    st.dataframe(
        filtered[display_cols].rename(columns=rename_map).round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # ── Product-level recommendation ─────────────────────────────────────────
    st.subheader("🔍 What else do customers buy with …?")
    all_products = sorted(set(rules_df["antecedents_clean"].tolist()))
    selected_product = st.selectbox("Select a product", all_products)

    if selected_product:
        recs = rules_df[rules_df["antecedents_clean"] == selected_product] \
            .sort_values("lift", ascending=False).head(5)
        if recs.empty:
            st.info("No rules found for this product. Try lowering the Confidence filter.")
        else:
            st.markdown("**Top recommendations:**")
            for _, row in recs.iterrows():
                cons = row["consequents_clean"]
                conf = float(row["confidence"])
                lift = float(row["lift"])
                st.markdown(f"""
                <div class='cogni-card'>
                    <span class='rule-pill'>{selected_product}</span>
                    → <span class='rule-pill'>{cons}</span>
                    &nbsp;&nbsp;
                    <span style='color:#94a3b8;font-size:0.8rem'>Confidence: <b>{conf:.0%}</b> &nbsp;|&nbsp; Lift: <b>{lift:.2f}</b></span>
                </div>
                """, unsafe_allow_html=True)

    # ── Rule statistics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Rule Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rules",      f"{len(rules_df):,}")
    c2.metric("Avg Confidence",   f"{rules_df['confidence'].mean():.2%}")
    c3.metric("Avg Lift",         f"{rules_df['lift'].mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segment Analysis")

    if rfm_df is None:
        st.error("RFM file not found. Run Week 4 notebook first.")
        st.stop()

    # Rename typo column if present
    if "Monetory" in rfm_df.columns:
        rfm_df = rfm_df.rename(columns={"Monetory": "Monetary"})

    # ── Segment distribution ─────────────────────────────────────────────────
    if "Segment" in rfm_df.columns:
        seg_counts = rfm_df["Segment"].value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Segment Breakdown")
            for seg, cnt in seg_counts.items():
                pct = cnt / len(rfm_df) * 100
                badge_cls = f"seg-{seg}" if seg in ["Premium", "Regular", "Budget", "Inactive"] else "seg-Regular"
                st.markdown(f"""
                <div class='cogni-card'>
                    <span class='seg-badge {badge_cls}'>{seg}</span>
                    <span style='float:right;font-size:1.3rem;font-weight:700'>{cnt}</span>
                    <div style='margin-top:8px;background:#334155;border-radius:4px;height:6px'>
                        <div style='width:{pct:.1f}%;background:#2dd4bf;height:6px;border-radius:4px'></div>
                    </div>
                    <div style='color:#64748b;font-size:0.75rem;margin-top:4px'>{pct:.1f}% of customers</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("RFM Averages by Segment")
            rfm_cols = [c for c in ["Recency", "Frequency", "Monetary"] if c in rfm_df.columns]
            if rfm_cols:
                seg_avg = rfm_df.groupby("Segment")[rfm_cols].mean().round(1)
                st.dataframe(seg_avg, use_container_width=True)

    st.markdown("---")

    # ── Customer lookup ───────────────────────────────────────────────────────
    st.subheader("🔍 Look up a Customer")
    cust_id = st.text_input("Enter Customer ID (e.g. C0001)", "")

    if cust_id:
        row = rfm_df[rfm_df["Customer_ID"] == cust_id]
        if row.empty:
            st.warning(f"Customer **{cust_id}** not found.")
        else:
            row = row.iloc[0]
            seg = row.get("Segment", "Unknown")
            badge_cls = f"seg-{seg}" if seg in ["Premium","Regular","Budget","Inactive"] else "seg-Regular"
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segment",   seg)
            if "Recency"   in row: c2.metric("Recency (days)", int(row["Recency"]))
            if "Frequency" in row: c3.metric("Frequency",      int(row["Frequency"]))
            if "Monetary"  in row: c4.metric("Monetary",       f"₹{row['Monetary']:,.0f}")

            # What did this customer buy?
            if df is not None:
                cust_tx = df[df["Customer_ID"] == cust_id]
                if not cust_tx.empty:
                    st.markdown("**Recent purchases:**")
                    prod_counts = cust_tx["Product"].value_counts().head(8)
                    pills = " ".join([f"<span class='rule-pill'>{p} ×{c}</span>" for p, c in prod_counts.items()])
                    st.markdown(pills, unsafe_allow_html=True)

    st.markdown("---")

    # ── RFM scatter ───────────────────────────────────────────────────────────
    st.subheader("📊 RFM Distribution")
    if "Frequency" in rfm_df.columns and "Monetary" in rfm_df.columns:
        st.scatter_chart(
            rfm_df.rename(columns={"Frequency": "x", "Monetary": "y"})[["x", "y"]],
            x="x", y="y"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – BERT SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 BERT Similarity":
    st.title("🤖 BERT Product Similarity")
    st.markdown("Uses Sentence-BERT embeddings (384-dim) + cosine similarity to find semantically similar products.")

    if sim_df is None:
        st.error("Similarity matrix not found. Run Week 6 notebook first.")
        st.stop()

    products_list = list(sim_df.index)
    selected = st.selectbox("Select a product", sorted(products_list))

    top_k = st.slider("Number of similar products", 3, 10, 5)

    if selected:
        scores = sim_df[selected].drop(selected).sort_values(ascending=False).head(top_k)

        st.subheader(f"Products most similar to **{selected}**")
        for prod, score in scores.items():
            bar_pct = int(score * 100)
            st.markdown(f"""
            <div class='cogni-card'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span class='rule-pill'>{prod}</span>
                    <span style='color:#2dd4bf;font-weight:700;font-family:JetBrains Mono'>{score:.3f}</span>
                </div>
                <div style='margin-top:10px;background:#334155;border-radius:4px;height:6px'>
                    <div style='width:{bar_pct}%;background:#2dd4bf;height:6px;border-radius:4px'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Semantic clusters
    if clusters is not None:
        st.markdown("---")
        st.subheader("🗂️ Semantic Product Clusters (K-Means on BERT embeddings)")
        for cid in sorted(clusters["SemanticCluster"].unique()):
            prods = clusters[clusters["SemanticCluster"] == cid]["Product"].tolist()
            pills = " ".join([f"<span class='rule-pill'>{p}</span>" for p in prods])
            st.markdown(f"**Cluster {cid}:** {pills}", unsafe_allow_html=True)
            st.markdown("")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – ASK COGNICART (RAG)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💬 Ask CogniCart (RAG)":
    st.title("💬 Ask CogniCart")
    st.markdown("Chat with your store data using the RAG pipeline built in Week 8.")

    st.info("💡 **To use real LLM responses:** open `week8.ipynb`, run all cells, then copy the `vectorstore` and `rag_chain` objects or use the exported CSV for rule-based fallback below.", icon="ℹ️")

    # ── Example questions ────────────────────────────────────────────────────
    st.subheader("💡 Try these questions")
    examples = [
        "What should I recommend to a customer who buys Milk?",
        "Which products are popular among Premium segment customers?",
        "Summarise buying behaviour of Budget segment customers.",
        "What goes well with Rice and Dal?",
        "Which customer segment spends the most?",
    ]
    cols = st.columns(len(examples))
    selected_example = None
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(ex[:40] + "…" if len(ex) > 40 else ex, key=f"ex_{i}"):
                selected_example = ex

    # ── Chat interface ────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Ask anything about your store data…",
                              value=selected_example or "",
                              key="rag_input")

    if st.button("Ask →", key="ask_btn") and question.strip():
        # Rule-based fallback (works without an LLM key)
        answer = rule_based_rag(question, rules_df, rfm_df)
        st.session_state.chat_history.append({"q": question, "a": answer})

    # Render chat history
    for turn in reversed(st.session_state.chat_history):
        st.markdown(f"<div class='chat-label'>You</div><div class='chat-user'>{turn['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-label'>CogniCart</div><div class='chat-cogni'>{turn['a']}</div>", unsafe_allow_html=True)
        st.markdown("")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.8rem'>
    ℹ️ <b>How to connect real LLM:</b> Replace <code>rule_based_rag()</code> in this file
    with your <code>ask_cognicart()</code> function from Week 8. 
    The function signature is identical — just swap the call.
    </div>
    """, unsafe_allow_html=True)

