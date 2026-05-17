"""
╔══════════════════════════════════════════════════════════════╗
║       LLM-Powered Autonomous Data Analyst                    ║
║       Author  : Saumaya Dube                                 ║
║       College : Rama University, Kanpur — UP 209217          ║
║       Guide   : Prof. (Dr.) C. S. Raghuvanshi               ║
║       Project : B.Tech Capstone 2025-2026                    ║
╚══════════════════════════════════════════════════════════════╝
HOW TO RUN:
    1. Add your FREE Groq API key in .env  →  GROQ_API_KEY=gsk_...
    2. Run:  streamlit run app.py
    3. Open: http://localhost:8501
"""
import os, re, sys, time, warnings
import pandas as pd
import numpy as np
import streamlit as st
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ingestion.profiler import load_csv, profile_dataframe, profile_to_text
from tools.analytics    import run_tool
from synthesis.agent    import plan_analysis, synthesize_results
from report.generator   import generate_pdf_report

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Data Analyst — Saumaya Dube",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
html,body,[class*="css"]{font-family:"Segoe UI",Arial,sans-serif}
.ada-header{background:linear-gradient(135deg,#1E2761 0%,#2B5BA8 60%,#4A90D9 100%);padding:28px 32px 22px;border-radius:14px;margin-bottom:22px}
.ada-header h1{color:#FFFFFF!important;font-size:1.9rem;font-weight:800;margin:0 0 6px}
.ada-header p{color:#BDD7EE!important;font-size:.97rem;margin:0}
.ada-metric{background:#FFFFFF;border:1.5px solid #DBEAFE;border-radius:12px;padding:16px 12px;text-align:center;box-shadow:0 2px 10px rgba(30,39,97,.08)}
.ada-metric .val{font-size:2rem;font-weight:800;color:#1E2761!important;line-height:1.1}
.ada-metric .lbl{font-size:.78rem;color:#64748B!important;margin-top:5px;font-weight:500}
.ada-feat{background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:12px;padding:18px 14px;text-align:center;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.ada-feat .icon{font-size:1.8rem;margin-bottom:8px}
.ada-feat .title{font-size:.92rem;font-weight:700;color:#1E2761!important}
.ada-feat .desc{font-size:.78rem;color:#64748B!important;margin-top:4px}
.tool-row{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;margin:5px 0;font-size:.93rem;font-weight:500}
.tool-row.pending{background:#EFF6FF;border-left:4px solid #4472C4;color:#1E40AF!important}
.tool-row.done{background:#F0FDF4;border-left:4px solid #16A34A;color:#14532D!important}
.tool-row.error{background:#FEF2F2;border-left:4px solid #DC2626;color:#7F1D1D!important}
.tool-row .tname{font-weight:700}
.report-wrap{background:#FFFFFF;border:1.5px solid #CBD5E1;border-radius:14px;padding:28px 32px;margin-top:8px;box-shadow:0 3px 14px rgba(0,0,0,.06)}
.report-wrap h2{font-size:1.1rem;font-weight:800;color:#1E2761!important;margin:22px 0 8px;padding-bottom:6px;border-bottom:2px solid #DBEAFE}
.report-wrap p{font-size:.95rem;color:#1E293B!important;line-height:1.75;margin:6px 0}
.report-wrap ul{padding-left:22px;margin:6px 0 10px}
.report-wrap li{font-size:.94rem;color:#1E293B!important;line-height:1.65;margin-bottom:5px}
.report-wrap b,.report-wrap strong{color:#1E2761!important;font-weight:700}
.step-badge{display:inline-block;background:#EFF6FF;color:#1E2761!important;border:1.5px solid #BFDBFE;border-radius:20px;padding:4px 14px;font-size:.82rem;font-weight:600;margin-bottom:6px}
/* Dataset selector buttons */
div[data-testid="stButton"] button[kind="secondary"] {
    background: #F8FAFF !important;
    border: 1.5px solid #DBEAFE !important;
    color: #1E2761 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 6px 4px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #DBEAFE !important;
    border-color: #4472C4 !important;
}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def render_narrative(text: str) -> str:
    html = ['<div class="report-wrap">']
    sections = re.split(r'(##[^\n]+)', text)
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if part.startswith("##"):
            heading = re.sub(r"^#+\s*", "", part).strip()
            html.append(f"<h2>{heading}</h2>")
            continue
        in_list = False
        for raw in part.split("\n"):
            line = raw.strip()
            if not line:
                if in_list:
                    html.append("</ul>"); in_list = False
                continue
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            if line.startswith(("- ", "* ", "• ")):
                if not in_list:
                    html.append("<ul>"); in_list = True
                html.append(f"<li>{re.sub(r'^[-*•]\\s+','',line)}</li>")
            else:
                if in_list:
                    html.append("</ul>"); in_list = False
                html.append(f"<p>{line}</p>")
        if in_list:
            html.append("</ul>")
    html.append("</div>")
    return "\n".join(html)


def render_log(rows: list) -> str:
    icons = {"pending": "⏳", "done": "✅", "error": "❌"}
    return "\n".join(
        f'<div class="tool-row {s}"><span>{icons[s]}</span>'
        f'<span class="tname">{n.replace("_"," ").title()}</span>'
        f'<span style="opacity:.85">{m}</span></div>'
        for s, n, m in rows
    )


def make_sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "PassengerId": range(1, n+1),
        "Survived":    rng.integers(0, 2, n),
        "Pclass":      rng.choice([1, 2, 3], n),
        "Sex":         rng.choice(["male","female"], n),
        "Age":         np.where(rng.random(n)>.15, rng.normal(30,13,n).clip(1,80), np.nan),
        "SibSp":       rng.integers(0, 6, n),
        "Parch":       rng.integers(0, 5, n),
        "Fare":        rng.exponential(32, n).round(2),
        "Embarked":    rng.choice(["S","C","Q",None], n, p=[0.70,0.20,0.09,0.01]),
    })


def api_key_ok() -> bool:
    k = os.environ.get("GROQ_API_KEY","").strip()
    return bool(k) and k != "your_groq_api_key_here" and len(k) > 15


# ── Restore API key from session_state on every rerun ────────
# Streamlit reruns the whole script on every interaction.
# Saving the key to session_state means it survives reruns.
if "groq_api_key" in st.session_state and st.session_state["groq_api_key"]:
    os.environ["GROQ_API_KEY"] = st.session_state["groq_api_key"]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Data Analyst AI")
    st.markdown("*Capstone — Rama University*")
    st.divider()
    st.markdown("### 🔑 Groq API Key")

    # Use session_state value as default so the box stays filled on rerun
    groq_key = st.text_input(
        "FREE key",
        type="password",
        placeholder="gsk_...",
        help="https://console.groq.com",
        label_visibility="collapsed",
        value=st.session_state.get("groq_api_key", ""),
    )
    if groq_key.strip():
        # Save to BOTH os.environ AND session_state
        os.environ["GROQ_API_KEY"]     = groq_key.strip()
        st.session_state["groq_api_key"] = groq_key.strip()

    # Show green tick if key is set, red cross if not
    if api_key_ok():
        st.success("✅ API key saved", icon="✅")
    else:
        st.error("❌ Key not set yet", icon="🔑")

    st.caption("[👉 Get FREE key — console.groq.com](https://console.groq.com)")
    st.divider()
    st.markdown("### 📌 Quick Guide")
    st.markdown("1. Paste **Groq key** above\n2. Upload **CSV file**\n3. *(Optional)* type goal\n4. Click **▶ Run Analysis**\n5. Download **PDF report**")
    st.divider()
    st.markdown("**Student:** Saumaya Dube")
    st.markdown("**College:** Rama University")
    st.markdown("**Guide:** Prof.(Dr.) C.S. Raghuvanshi")


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="ada-header">
  <h1>📊 LLM-Powered Autonomous Data Analyst</h1>
  <p>Upload any CSV → AI automatically analyzes → Complete business report in minutes &nbsp;|&nbsp;
     Saumaya Dube &nbsp;|&nbsp; Rama University, Kanpur</p>
</div>""", unsafe_allow_html=True)


# ── Upload + Goal ─────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("#### 📁 Upload Dataset")
    uploaded_file = st.file_uploader("CSV file", type=["csv"],
                                      label_visibility="collapsed")
with c2:
    st.markdown("#### 🎯 Analysis Goal *(optional)*")
    user_goal = st.text_area("goal",
        placeholder="e.g. 'What factors affect passenger survival?'",
        height=104, label_visibility="collapsed")

st.markdown("")
# ── Dataset selector ─────────────────────────────────────────
DATASETS = {
    "titanic":           ("🚢 Titanic Survival",      "Demographics",   "891 rows × 11 cols"),
    "heart_disease":     ("❤️ Heart Disease",          "Healthcare",     "303 rows × 14 cols"),
    "house_prices":      ("🏠 House Prices",           "Real Estate",    "1460 rows × 17 cols"),
    "bike_sharing":      ("🚲 Bike Sharing",           "Transport",      "500 rows × 15 cols"),
    "customer_churn":    ("📱 Customer Churn",         "Telecom",        "500 rows × 14 cols"),
    "wine_quality":      ("🍷 Wine Quality",           "Food Science",   "500 rows × 13 cols"),
    "ibm_hr_analytics":  ("👔 IBM HR Analytics",       "HR",             "500 rows × 23 cols"),
    "covid19_country":   ("🦠 COVID-19 Country Stats", "Public Health",  "86 rows × 16 cols"),
    "nyc_taxi":          ("🚕 NYC Taxi Trips",          "Transportation", "500 rows × 19 cols"),
    "ecommerce_orders":  ("🛒 E-commerce Orders",      "E-commerce",     "500 rows × 21 cols"),
    "credit_card_fraud": ("💳 Credit Card Fraud",      "Finance",        "500 rows × 31 cols"),
    "rossmann_sales":    ("🏪 Rossmann Store Sales",   "Retail",         "500 rows × 17 cols"),
}

st.markdown("#### 📂 Or Load One of 12 Sample Datasets")

# Show dataset cards in a 4-column grid
d_cols = st.columns(4, gap="small")
for idx, (key, (label, domain, size)) in enumerate(DATASETS.items()):
    with d_cols[idx % 4]:
        btn_label = f"{label}"
        if st.button(btn_label, key=f"ds_{key}", use_container_width=True,
                     help=f"Domain: {domain}  |  Size: {size}"):
            csv_path = os.path.join(os.path.dirname(__file__), "data", "sample", f"{key}.csv")
            try:
                sdf = pd.read_csv(csv_path)
                st.session_state["sample_df"] = sdf
                st.session_state["sample_name"] = f"{key}.csv"
                st.success(f"✅ Loaded **{label}** — {sdf.shape[0]} rows × {sdf.shape[1]} cols")
            except Exception as e:
                st.error(f"Could not load {key}: {e}")

st.markdown("")
_, cr = st.columns([3, 1], gap="large")
with cr:
    run_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)

st.divider()


# ── Landing (no run yet) ─────────────────────────────────────
if not run_clicked:
    st.markdown("### ✨ What This System Does Automatically")
    feats = [
        ("📊","Descriptive Stats","Mean, median, std, skewness & kurtosis"),
        ("🔍","Missing Values","Counts, %, severity classification"),
        ("🔗","Correlation","Pearson matrix & top correlations"),
        ("⚠️","Outlier Detection","IQR method — box plots per column"),
        ("📈","Distribution","Shapiro-Wilk normality + Q-Q plots"),
        ("🧪","Hypothesis Testing","t-test, ANOVA, chi-square — auto-selected"),
        ("🎯","Feature Importance","Mutual information ranking"),
        ("📝","AI Business Report","LLaMA 3 writes the full narrative"),
    ]
    fcols = st.columns(4, gap="small")
    for i,(ico,ttl,dsc) in enumerate(feats):
        with fcols[i%4]:
            st.markdown(
                f'<div class="ada-feat"><div class="icon">{ico}</div>'
                f'<div class="title">{ttl}</div><div class="desc">{dsc}</div></div>',
                unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════

# Resolve dataframe
df = None
dataset_name = "dataset.csv"
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        dataset_name = uploaded_file.name
    except Exception as exc:
        st.error(f"❌ Could not read file: {exc}"); st.stop()
elif "sample_df" in st.session_state:
    df = st.session_state["sample_df"]
    dataset_name = st.session_state.get("sample_name", "sample_dataset.csv")
else:
    st.warning("⚠️ Upload a CSV file first, or click **Load Sample Dataset**."); st.stop()

# API key
if not api_key_ok():
    st.error("**❌ Groq API Key Required**\n\n"
             "1. Go to https://console.groq.com (free)\n"
             "2. Create API Key\n"
             "3. Paste in the sidebar"); st.stop()

# ── Step 1: Profile ───────────────────────────────────────────
st.markdown('<div class="step-badge">Step 1 / 4 — Profiling Dataset</div>', unsafe_allow_html=True)
prog = st.progress(0, text="Reading dataset…")
try:
    profile      = profile_dataframe(df)
    profile_text = profile_to_text(profile)
except Exception as exc:
    st.error(f"❌ Profiling failed: {exc}"); st.stop()

prog.progress(15, text="Dataset profiled ✓")

m1,m2,m3,m4 = st.columns(4, gap="small")
for col,val,lbl in [
    (m1, f"{profile['shape']['rows']:,}",    "Rows"),
    (m2, str(profile['shape']['columns']),   "Columns"),
    (m3, str(profile['missing_total']),       "Missing Cells"),
    (m4, str(profile['duplicate_rows']),      "Duplicate Rows"),
]:
    with col:
        st.markdown(
            f'<div class="ada-metric"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True)
st.markdown("")

# ── Step 2: Plan ──────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 2 / 4 — AI Planning</div>', unsafe_allow_html=True)
prog.progress(25, text="LLM is planning…")
plan = plan_analysis(profile_text, user_goal)
tools_to_run = plan.get("tools_to_run") or []

# Show a clear warning if LLM planning failed (with the real reason)
if plan.get("_error"):
    st.warning(
        f"⚠️ **AI planning unavailable — using default tool set.**\n\n"
        f"**Reason:** {plan['_error']}\n\n"
        f"**Fix:** Make sure your Groq API key is correct in the sidebar. "
        f"Get a FREE key at https://console.groq.com"
    )
else:
    st.info("🤖 **AI Plan:** Running " + ", ".join(f"`{t}`" for t in tools_to_run))
    if plan.get("reasoning"):
        with st.expander("📋 Why these tools?", expanded=False):
            st.write(plan["reasoning"])

# ── Step 3: Tools ─────────────────────────────────────────────
st.markdown('<div class="step-badge">Step 3 / 4 — Running Tools</div>', unsafe_allow_html=True)
st.markdown("**🔬 Tool Execution Log**")
log_ph = st.empty()
log_rows, tool_summaries, charts = [], [], {}
RANGE = (30, 78)
n_tools = len(tools_to_run)

for idx, tname in enumerate(tools_to_run):
    pct = RANGE[0] + int((idx/n_tools)*(RANGE[1]-RANGE[0]))
    prog.progress(pct, text=f"Running {tname} ({idx+1}/{n_tools})…")
    log_rows.append(("pending", tname, "running…"))
    log_ph.markdown(render_log(log_rows), unsafe_allow_html=True)

    result = run_tool(tname, df)
    log_rows[-1] = (
        "error" if result.get("error") else "done",
        tname,
        str(result.get("error",""))[:100] if result.get("error") else "completed successfully"
    )
    log_ph.markdown(render_log(log_rows), unsafe_allow_html=True)
    if result.get("summary"): tool_summaries.append(result["summary"])
    if result.get("chart"):   charts[tname] = result["chart"]

prog.progress(80, text="Synthesizing insights…")

# ── Step 4: Synthesize ────────────────────────────────────────
st.markdown('<div class="step-badge">Step 4 / 4 — AI Writing Report</div>', unsafe_allow_html=True)
try:
    narrative = synthesize_results(profile_text, tool_summaries, user_goal)
except Exception as exc:
    narrative = (
        f"## 📊 Executive Summary\nAnalysis of **{dataset_name}** — "
        f"{profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns.\n\n"
        f"## 🔍 Tool Findings\n" + "\n\n".join(tool_summaries[:4])
        + f"\n\n*(LLM unavailable: {exc})*"
    )

prog.progress(100, text="✅ Done!")
time.sleep(0.3); prog.empty()
st.success(f"✅ Done! Ran **{len(tools_to_run)} tools**, made **{len(charts)} charts**, wrote AI report.")

# ════════════════════════════════════════════════════════════
# RESULTS TABS
# ════════════════════════════════════════════════════════════
st.markdown("## 📋 Results")
tab1, tab2, tab3, tab4 = st.tabs(["📝 AI Report","📊 Charts","📈 Data Preview","⬇️ Download"])

with tab1:
    st.markdown("#### 🤖 AI-Generated Business Report")
    st.markdown(render_narrative(narrative), unsafe_allow_html=True)

with tab2:
    st.markdown("#### 📊 Visualizations")
    if charts:
        for tname, b64 in charts.items():
            label = tname.replace("_"," ").title()
            with st.expander(f"📈 {label}", expanded=True):
                # ✅ use width= param (replaces old deprecated param)
                st.image(f"data:image/png;base64,{b64}", caption=label, width=900)
    else:
        st.info("No charts generated.")

with tab3:
    st.markdown("#### 🗂️ Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
    st.markdown("#### 🏷️ Column Information")
    col_info = pd.DataFrame(profile["columns"])
    # ✅ Fixed: stringify sample_values to avoid PyArrow mixed-type error
    col_info["sample_values"] = col_info["sample_values"].apply(
        lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else str(x))
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    if profile.get("numeric_summary"):
        st.markdown("#### 📐 Numeric Statistics")
        sdf = pd.DataFrame(profile["numeric_summary"]).T.reset_index().rename(columns={"index":"column"})
        st.dataframe(sdf, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("#### ⬇️ Export Results")
    with st.spinner("Building PDF…"):
        try:
            pdf_bytes = generate_pdf_report(dataset_name, profile, narrative, tool_summaries)
            st.download_button("📄 Download PDF Report", pdf_bytes,
                               f"report_{dataset_name.replace('.csv','')}.pdf",
                               "application/pdf", type="primary", use_container_width=True)
        except Exception as exc:
            st.error(f"PDF failed: {exc}")
    st.markdown("")
    st.download_button("📊 Download CSV", df.to_csv(index=False).encode(),
                       f"cleaned_{dataset_name}", "text/csv", use_container_width=True)
    st.markdown("")
    st.download_button("📝 Download AI Report (txt)", narrative.encode(),
                       f"report_{dataset_name.replace('.csv','')}.txt",
                       "text/plain", use_container_width=True)
    st.divider()
    st.caption("📌 LLM-Powered Autonomous Data Analyst | Saumaya Dube | "
               "Rama University, Kanpur | Prof.(Dr.) C.S. Raghuvanshi | 2025-2026")
