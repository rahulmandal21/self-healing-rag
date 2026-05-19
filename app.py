import os
import streamlit as st
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
from ingest import ingest, DB_PATH
if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
    ingest()
import pandas as pd
from graph import graph

st.set_page_config(
    page_title="Self-Healing RAG Pipeline",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg: #080c10;
        --surface: #0d1117;
        --surface2: #111820;
        --surface3: #161e28;
        --border: rgba(255,255,255,0.06);
        --border2: rgba(255,255,255,0.1);
        --text: #e8edf2;
        --muted: #5a6a7a;
        --accent: #00d4ff;
        --accent2: #7c3aed;
        --green: #00e5a0;
        --red: #ff4d6d;
        --amber: #fbbf24;
    }

    html, body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }

    .stApp { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; color: var(--text); }
    .main { background: var(--bg) !important; }
    .main .block-container { padding: 0 !important; max-width: 100% !important; }

    #MainMenu, footer, header { visibility: hidden; }
    div[data-testid="stDecoration"] { display: none; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        width: 280px !important;
    }
    section[data-testid="stSidebar"] > div { padding: 0 !important; }

    .sb-head {
        padding: 28px 20px 20px;
        border-bottom: 1px solid var(--border);
    }
    .sb-brand {
        display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
    }
    .sb-brand-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #00d4ff22, #7c3aed33);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
    }
    .sb-brand-name {
        font-family: 'Syne', sans-serif;
        font-weight: 700; font-size: 14px; color: var(--text); letter-spacing: 0.01em;
    }
    .sb-brand-sub {
        font-size: 11px; color: var(--muted);
        font-family: 'DM Mono', monospace; letter-spacing: 0.03em;
    }

    .sb-stat-grid {
        display: grid; grid-template-columns: 1fr 1fr 1fr;
        gap: 8px; padding: 16px 20px;
        border-bottom: 1px solid var(--border);
    }
    .sb-stat-box {
        background: var(--surface2); border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 8px; text-align: center;
    }
    .sb-stat-box.featured {
        grid-column: 1 / -1;
        background: linear-gradient(135deg, #00d4ff0a, #7c3aed0a);
        border-color: rgba(0,212,255,0.15);
    }
    .sb-stat-num {
        font-family: 'Syne', sans-serif; font-weight: 800;
        font-size: 28px; color: var(--accent); line-height: 1;
    }
    .sb-stat-num.sm { font-size: 22px; color: #8a9ab0; }
    .sb-stat-num.green { font-size: 22px; color: var(--green); }
    .sb-stat-num.red { font-size: 22px; color: var(--red); }
    .sb-stat-lbl {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.12em;
        color: var(--muted); margin-top: 4px; font-family: 'DM Mono', monospace;
    }

    .sb-q-section { padding: 14px 20px; }
    .sb-section-label {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em;
        color: var(--muted); font-family: 'DM Mono', monospace; margin-bottom: 10px;
    }
    .sb-q-item {
        display: flex; align-items: center; gap: 8px;
        padding: 7px 0; border-bottom: 1px solid var(--border);
        font-size: 11.5px; color: #8a9ab0; line-height: 1.4;
    }
    .sb-q-check {
        width: 16px; height: 16px;
        background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.3);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 8px; color: var(--green);
    }
    .sb-q-fail {
        width: 16px; height: 16px;
        background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.3);
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        flex-shrink: 0; font-size: 8px; color: var(--red);
    }

    .sb-stack-section {
        padding: 14px 20px;
        border-top: 1px solid var(--border);
    }
    .sb-stack-item {
        display: flex; align-items: center; gap: 8px;
        padding: 4px 0; font-size: 11.5px; color: #6a7d90;
        font-family: 'DM Mono', monospace;
    }
    .sb-stack-dot {
        width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
    }

    /* ── MAIN HEADER ── */
    .main-header { padding: 10px 36px 0; }

    .page-title {
        font-family: 'Space Grotesk', sans-serif; font-weight: 800;
        font-size: 28px; color: var(--text);
        letter-spacing: -0.02em; line-height: 1.1;
    }
    .page-title span {
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .page-sub {
        font-size: 13px; color: var(--muted); margin-top: 5px;
        font-family: 'DM Mono', monospace;
    }
    .badge-row { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    .badge {
        font-size: 10px; font-family: 'DM Mono', monospace;
        letter-spacing: 0.05em; padding: 4px 10px;
        border-radius: 99px; border: 1px solid var(--border2);
        color: var(--muted); background: var(--surface2);
    }
    .badge.cyan  { border-color: rgba(0,212,255,0.25);  color: var(--accent);  background: rgba(0,212,255,0.05); }
    .badge.purple{ border-color: rgba(124,58,237,0.3);  color: #a78bfa;        background: rgba(124,58,237,0.06); }
    .badge.green { border-color: rgba(0,229,160,0.25);  color: var(--green);   background: rgba(0,229,160,0.05); }

    /* ── PIPELINE ── */
    .pipeline-row {
        display: flex; align-items: center;
        margin: 0 36px 16px; gap: 0;
        padding: 16px 20px;
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 14px; overflow-x: auto;
    }
    .pipe-step { display: flex; flex-direction: column; align-items: center; gap: 6px; flex-shrink: 0; }
    .pipe-node {
        width: 40px; height: 40px; border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; border: 1px solid var(--border2);
        background: var(--surface3);
    }
    .pipe-node.active {
        background: rgba(0,212,255,0.1); border-color: rgba(0,212,255,0.3);
        box-shadow: 0 0 12px rgba(0,212,255,0.1);
    }
    .pipe-label {
        font-size: 9px; color: var(--muted);
        font-family: 'DM Mono', monospace; text-align: center; white-space: nowrap;
    }
    .pipe-arrow {
        width: 28px; height: 1px; background: var(--border2); flex-shrink: 0;
        position: relative; margin: 0 2px;
    }
    .pipe-arrow::after {
        content: '›'; position: absolute; right: -6px; top: -8px;
        color: var(--border2); font-size: 16px;
    }

    /* ── INPUT ── */
    div[data-testid="stTextInput"] label { display: none !important; }
    div[data-testid="stTextInput"] > div { background: transparent !important; }
    div[data-testid="stTextInput"] input {
        background: var(--surface) !important; border: 1px solid var(--border2) !important;
        color: var(--text) !important; border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important; font-size: 14px !important;
        padding: 12px 20px !important; transition: all 0.2s !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(0,212,255,0.4) !important;
        box-shadow: 0 0 0 3px rgba(0,212,255,0.05) !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

    div[data-testid="stButton"] button {
        border-radius: 10px !important; font-weight: 700 !important;
        font-family: 'Syne', sans-serif !important; font-size: 13px !important;
        transition: all 0.2s ease !important; letter-spacing: 0.02em !important;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff, #0090cc) !important;
        border: none !important; color: #001f2e !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        opacity: 0.9 !important; transform: translateY(-1px) !important;
    }
    div[data-testid="stButton"] button:not([kind]),
    div[data-testid="stButton"] button[kind="secondary"] {
        background: var(--surface3) !important; border: 1px solid var(--border2) !important;
        color: var(--muted) !important;
    }
    div[data-testid="stButton"] button:not([kind]):hover {
        border-color: rgba(255,77,109,0.4) !important; color: var(--red) !important;
    }

    div[data-testid="stSpinner"] p {
        color: var(--muted) !important; font-family: 'DM Mono', monospace !important; font-size: 13px !important;
    }

    /* ── ANSWER ── */
    .answer-card {
        position: relative; background: var(--surface);
        border: 1px solid var(--border2); border-radius: 16px;
        padding: 24px; margin-bottom: 16px; overflow: hidden;
    }
    .answer-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent2), transparent);
    }
    .answer-tag {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.15em;
        color: var(--accent); font-family: 'DM Mono', monospace;
        margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
    }
    .answer-tag::after { content: ''; flex: 1; height: 1px; background: rgba(0,212,255,0.12); }
    .answer-text { font-size: 15px; line-height: 1.75; color: #cdd5e0; }

    /* ── STATUS ROW ── */
    .status-row {
        display: grid; grid-template-columns: 1fr 1fr;
        gap: 12px; margin-bottom: 16px;
    }
    .status-pill {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 14px 18px;
        display: flex; align-items: center; gap: 12px;
    }
    .status-icon {
        width: 36px; height: 36px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; flex-shrink: 0;
    }
    .status-icon.pass { background: rgba(0,229,160,0.1);  border: 1px solid rgba(0,229,160,0.2); }
    .status-icon.fail { background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.2); }
    .status-icon.iter { background: rgba(0,212,255,0.1);  border: 1px solid rgba(0,212,255,0.2); }
    .status-label {
        font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
        font-family: 'DM Mono', monospace; margin-bottom: 2px;
    }
    .status-label.pass { color: var(--green); }
    .status-label.fail { color: var(--red); }
    .status-label.iter { color: var(--accent); }
    .status-val { font-size: 13px; color: #8a9ab0; line-height: 1.4; }

    /* ── DETAIL CARD ── */
    .detail-card {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 20px 24px;
    }
    .detail-title {
        font-size: 9px; text-transform: uppercase; letter-spacing: 0.14em;
        color: var(--muted); font-family: 'DM Mono', monospace; margin-bottom: 16px;
    }
    .detail-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px;
    }
    .detail-row:last-child { border-bottom: none; }
    .detail-key { color: var(--muted); font-family: 'DM Mono', monospace; font-size: 12px; }
    .detail-val { color: var(--text); font-weight: 500; }
    .detail-val.accent { color: var(--accent); }
    .detail-val.green  { color: var(--green); }
    .detail-val.red    { color: var(--red); }

    /* ── EMPTY STATE ── */
    .empty-state {
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; padding: 80px 20px; text-align: center; gap: 16px;
    }
    .empty-icon {
        width: 72px; height: 72px; border-radius: 20px;
        background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.08));
        border: 1px solid rgba(0,212,255,0.12);
        display: flex; align-items: center; justify-content: center;
        font-size: 32px; animation: pulse 3s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,255,0.1); }
        50%       { box-shadow: 0 0 0 12px rgba(0,212,255,0); }
    }
    .empty-title {
        font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700;
        color: var(--text); opacity: 0.6;
    }
    .empty-sub { font-size: 13px; color: var(--muted); max-width: 320px; line-height: 1.6; }

    /* ── CONTENT WRAPPER ── */
    .content-wrap { padding: 0 36px 36px; }

    /* ── CHIPS ── */
    div[data-testid="stButton"] button.sample-btn {
        background: var(--surface) !important; border: 1px solid var(--border2) !important;
        color: #7a8fa8 !important; font-size: 11.5px !important;
        padding: 7px 14px !important; border-radius: 99px !important;
        font-family: 'DM Sans', sans-serif !important; font-weight: 400 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state
for k, v in [("result", None), ("last_query", ""), ("clear_count", 0), ("query_value", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# SIDEBAR

st.sidebar.markdown("""
<div class="sb-head">
    <div class="sb-brand">
        <div class="sb-brand-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="15" x2="8" y2="15"/><line x1="12" y1="15" x2="12" y2="15"/><line x1="16" y1="15" x2="16" y2="15"/></svg></div>
        <div class="sb-brand-name">Self-Healing RAG</div>
    </div>
    <div class="sb-brand-sub">infosys · annual report 2024–25</div>
</div>
""", unsafe_allow_html=True)

eval_df = None
try:
    eval_df = pd.read_csv("eval/eval_results.csv")
except Exception:
    pass

if eval_df is not None:
    pass_count = int((eval_df["critic_verdict"] == "pass").sum())
    fail_count = int(len(eval_df) - pass_count)

    st.sidebar.markdown(f"""
<div class="sb-stat-grid">
    <div class="sb-stat-box featured">
        <div class="sb-stat-num">{int(pass_count/len(eval_df)*100)}%</div>
        <div class="sb-stat-lbl">Critic Pass Rate</div>
    </div>
    <div class="sb-stat-box">
        <div class="sb-stat-num sm">{len(eval_df)}</div>
        <div class="sb-stat-lbl">Total</div>
    </div>
    <div class="sb-stat-box">
        <div class="sb-stat-num green">{pass_count}</div>
        <div class="sb-stat-lbl">Passed</div>
    </div>
    <div class="sb-stat-box">
        <div class="sb-stat-num red">{fail_count}</div>
        <div class="sb-stat-lbl">Failed</div>
    </div>
</div>
""", unsafe_allow_html=True)

    rows_html = ""
    for _, row in eval_df.iterrows():
        q = row["question"][:46] + "…" if len(row["question"]) > 46 else row["question"]
        icon_div = '<div class="sb-q-check">✓</div>' if row["critic_verdict"] == "pass" else '<div class="sb-q-fail">✗</div>'
        rows_html += f'<div class="sb-q-item">{icon_div}<span>{q}</span></div>'

    st.sidebar.markdown(f"""
<div class="sb-q-section">
    <div class="sb-section-label">Questions Tested</div>
    {rows_html}
</div>
""", unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
<div class="sb-stat-grid">
    <div class="sb-stat-box featured">
        <div class="sb-stat-num">—</div>
        <div class="sb-stat-lbl">Critic Pass Rate</div>
    </div>
</div>
<div class="sb-q-section">
    <div class="sb-section-label">Questions Tested</div>
    <div class="sb-q-item" style="color:#2e374d">Run eval first to see results.</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sb-stack-section">
    <div class="sb-section-label" style="margin-bottom:10px">Stack</div>
    <div class="sb-stack-item"><div class="sb-stack-dot" style="background:#00d4ff;opacity:0.5"></div> LangGraph</div>
    <div class="sb-stack-item"><div class="sb-stack-dot" style="background:#7c3aed;opacity:0.5"></div> Groq · llama-3.1-8b-instant</div>
    <div class="sb-stack-item"><div class="sb-stack-dot" style="background:#fbbf24;opacity:0.5"></div> ChromaDB · 1,104 chunks</div>
    <div class="sb-stack-item"><div class="sb-stack-dot" style="background:#00e5a0;opacity:0.5"></div> all-MiniLM-L6-v2</div>
</div>
""", unsafe_allow_html=True)


# MAIN — HEADER

st.markdown("""
<div class="main-header">
    <div class="page-title">Self-Healing <span>RAG Pipeline</span></div>
    <div class="page-sub">Infosys Integrated Annual Report 2024–25</div>
    <div class="badge-row">
        <span class="badge cyan">LangGraph</span>
        <span class="badge purple">Groq LLaMA 3.1</span>
        <span class="badge green">ChromaDB</span>
        <span class="badge">Critic Agent</span>
        <span class="badge">Self-Healing</span>
    </div>
</div>
""", unsafe_allow_html=True)


#  Search row 
st.markdown('<div style="padding: 0 36px 4px;">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([7, 1.4, 0.9])
with col1:
    query = st.text_input(
        "", placeholder="Ask anything about Infosys annual report...",
        label_visibility="collapsed",
        value=st.session_state.query_value,
        key=f"q_{st.session_state.clear_count}"
    )
with col2:
    run = st.button("Run Query →", use_container_width=True, type="primary")
with col3:
    if st.button("Clear", use_container_width=True):
        st.session_state.result = None
        st.session_state.last_query = ""
        st.session_state.query_value = ""
        st.session_state.clear_count += 1
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ── Suggestion chips ───────────────────────────────────────────
if "show_more_chips" not in st.session_state:
    st.session_state.show_more_chips = False

suggestions_main = [
    "What was total revenue in FY25?",
    "How many employees does Infosys have?",
    "What was the operating margin?",
    "Large deal TCV in FY2025?",
]
suggestions_extra = [
    "What was Infosys net profit in FY25?",
    "How many active clients does Infosys have?",
    "What is Infosys dividend per share?",
    "What was free cash flow in FY25?",
    "In how many countries does Infosys operate?",
    "What was basic EPS in FY2025?",
    "What are Infosys key business segments?",
    "What was headcount attrition rate?",
]

chip_cols = st.columns(len(suggestions_main) + 1)
for i, label in enumerate(suggestions_main):
    with chip_cols[i]:
        if st.button(label, key=f"chip_{i}", use_container_width=True):
            st.session_state.query_value = label
            st.session_state.clear_count += 1
            st.rerun()
with chip_cols[-1]:
    toggle_label = "▲ Less" if st.session_state.show_more_chips else "＋ More"
    if st.button(toggle_label, key="chip_toggle", use_container_width=True):
        st.session_state.show_more_chips = not st.session_state.show_more_chips
        st.rerun()

if st.session_state.show_more_chips:
    extra_cols = st.columns(4)
    for i, label in enumerate(suggestions_extra):
        with extra_cols[i % 4]:
            if st.button(label, key=f"chip_extra_{i}", use_container_width=True):
                st.session_state.query_value = label
                st.session_state.clear_count += 1
                st.rerun()

# Run pipeline 
if run and query:
    try:
        with st.spinner("Running pipeline..."):
            result = graph.invoke({"query": query, "iterations": 0})
        st.session_state.result = result
        st.session_state.last_query = query
    except Exception as e:
        err = str(e)
        if "rate_limit" in err or "429" in err:
            st.markdown("""
<div style="background:#100a0a;border:1px solid rgba(255,77,109,0.25);border-radius:12px;padding:16px 20px;margin:16px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.12em;color:#ff4d6d;font-family:'DM Mono',monospace;margin-bottom:6px;">⚠ Rate Limit Reached</div>
    <div style="font-size:13px;color:#8a9ab0;">Groq API limit hit. Please wait a few seconds and try again.</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div style="background:#100a0a;border:1px solid rgba(255,77,109,0.25);border-radius:12px;padding:16px 20px;margin:16px 36px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.12em;color:#ff4d6d;font-family:'DM Mono',monospace;margin-bottom:6px;">⚠ Pipeline Error</div>
    <div style="font-size:13px;color:#8a9ab0;">{err[:200]}</div>
</div>
""", unsafe_allow_html=True)

#  Results or empty state
st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

if st.session_state.result:
    result = st.session_state.result
    iters  = result.get("iterations", 0)

    st.markdown(f"""
<div class="answer-card">
    <div class="answer-tag">✦ Answer</div>
    <div class="answer-text">{result['answer']}</div>
</div>
""", unsafe_allow_html=True)

    healing     = "Yes" if iters > 0 else "No"
    healing_cls = "accent" if iters > 0 else ""
    verdict_up  = result["critic_verdict"].upper()
    verdict_cls = "green" if result["critic_verdict"] == "pass" else "red"

    if result["critic_verdict"] == "pass":
        critic_icon  = "✅"
        critic_icls  = "pass"
        critic_lcls  = "pass"
        critic_ltxt  = "Critic · PASS"
    else:
        critic_icon  = "❌"
        critic_icls  = "fail"
        critic_lcls  = "fail"
        critic_ltxt  = "Critic · FAIL"

    iter_txt = f"Self-healing triggered · {iters} reformulation(s)" if iters > 0 else "Answered on first attempt · 0 iterations"

    st.markdown(f"""
<div class="status-row">
    <div class="status-pill">
        <div class="status-icon {critic_icls}">{critic_icon}</div>
        <div>
            <div class="status-label {critic_lcls}">{critic_ltxt}</div>
            <div class="status-val">{result['critic_reason']}</div>
        </div>
    </div>
    <div class="status-pill">
        <div class="status-icon iter">⚡</div>
        <div>
            <div class="status-label iter">Self-Healing</div>
            <div class="status-val">{iter_txt}</div>
        </div>
    </div>
</div>

<div class="detail-card">
    <div class="detail-title">Pipeline Details</div>
    <div class="detail-row">
        <span class="detail-key">query</span>
        <span class="detail-val accent" style="max-width:60%;text-align:right;word-break:break-word;font-size:12px">{st.session_state.last_query[:60]}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">self-healing triggered</span>
        <span class="detail-val {healing_cls}">{healing}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">reformulations</span>
        <span class="detail-val">{iters}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">critic verdict</span>
        <span class="detail-val {verdict_cls}">{verdict_up}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">model</span>
        <span class="detail-val">llama-3.1-8b-instant</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">chunks retrieved</span>
        <span class="detail-val">5</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">vector store</span>
        <span class="detail-val">ChromaDB · 1,104 chunks</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">embeddings</span>
        <span class="detail-val">all-MiniLM-L6-v2</span>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="empty-state">
    <div class="empty-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="url(#g)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><defs><linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#00d4ff"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient></defs><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="15" x2="8" y2="15"/><line x1="12" y1="15" x2="12" y2="15"/><line x1="16" y1="15" x2="16" y2="15"/></svg></div>
    <div class="empty-title">Ready to answer</div>
    <div class="empty-sub">Ask anything from the Infosys Integrated Annual Report 2024–25. The critic agent will validate and self-heal if needed.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)