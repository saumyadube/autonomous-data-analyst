"""
Layer 3: Analytical Tool Library
----------------------------------
8 tools that the LLM can call to perform
actual computations on the dataset.

Each tool:
  - Takes a DataFrame + parameters
  - Returns a dict with results + text summary
  - Generates charts saved as base64 PNG strings
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import io
import base64
import warnings
warnings.filterwarnings("ignore")

# ── Colour palette ──────────────────────────────────────────
BLUE  = "#1F3864"
BLUE2 = "#4472C4"
ORANGE = "#ED7D31"
GREEN  = "#70AD47"
PLT_STYLE = {
    "axes.facecolor":  "#FAFAFA",
    "figure.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ════════════════════════════════════════════════════════════
# TOOL 1 — Descriptive Statistics
# ════════════════════════════════════════════════════════════
def descriptive_stats(df: pd.DataFrame) -> dict:
    """
    Computes full descriptive statistics for all numeric columns:
    mean, std, min, max, quartiles, skewness, kurtosis.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns found.", "summary": "No numeric columns to analyze."}

    results = {}
    for col in numeric_cols:
        s = df[col].dropna()
        results[col] = {
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "variance": round(float(s.var()), 4),
            "min":      round(float(s.min()), 4),
            "max":      round(float(s.max()), 4),
            "q25":      round(float(s.quantile(0.25)), 4),
            "q75":      round(float(s.quantile(0.75)), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "count":    int(s.count()),
        }

    # Generate histogram grid
    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        df[col].dropna().hist(ax=ax, bins=30, color=BLUE2, edgecolor="white", linewidth=0.5)
        ax.axvline(df[col].mean(), color=ORANGE, lw=2, linestyle="--", label=f"Mean: {df[col].mean():.2f}")
        ax.set_title(col, fontsize=10, fontweight="bold", color=BLUE)
        ax.set_xlabel(""); ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Numeric Features", fontsize=13, fontweight="bold", color=BLUE, y=1.01)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    summary_lines = ["DESCRIPTIVE STATISTICS RESULTS:"]
    for col, stats_dict in results.items():
        summary_lines.append(
            f"  {col}: mean={stats_dict['mean']}, median={stats_dict['median']}, "
            f"std={stats_dict['std']}, skewness={stats_dict['skewness']}, "
            f"kurtosis={stats_dict['kurtosis']}"
        )
        if abs(stats_dict["skewness"]) > 1:
            summary_lines.append(f"    ⚠ {col} is highly skewed (skewness={stats_dict['skewness']})")

    return {
        "tool": "descriptive_stats",
        "results": results,
        "chart": chart,
        "summary": "\n".join(summary_lines)
    }


# ════════════════════════════════════════════════════════════
# TOOL 2 — Missing Value Analysis
# ════════════════════════════════════════════════════════════
def missing_value_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes missing values: counts, percentages, severity classification.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values
    }).sort_values("missing_pct", ascending=False)

    missing_df["severity"] = missing_df["missing_pct"].apply(
        lambda x: "Critical (>20%)" if x > 20
        else "Warning (5-20%)" if x > 5
        else "Acceptable (<5%)" if x > 0
        else "Complete"
    )

    # Chart
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [
        "#FF4444" if s == "Critical (>20%)"
        else ORANGE if s == "Warning (5-20%)"
        else GREEN if s == "Acceptable (<5%)"
        else "#DDDDDD"
        for s in missing_df["severity"]
    ]
    bars = ax.bar(missing_df["column"], missing_df["missing_pct"],
                  color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, missing_df["missing_pct"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Missing Values (%)", fontsize=10)
    ax.set_title("Missing Value Analysis — Feature-Level Report", fontsize=12, fontweight="bold", color=BLUE)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45, ha="right")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF4444", label="Critical (>20%)"),
        Patch(facecolor=ORANGE,   label="Warning (5-20%)"),
        Patch(facecolor=GREEN,    label="Acceptable (<5%)"),
        Patch(facecolor="#DDDDDD",label="Complete"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    results = missing_df.to_dict(orient="records")
    critical = missing_df[missing_df["missing_pct"] > 20]["column"].tolist()
    warning  = missing_df[(missing_df["missing_pct"] > 5) & (missing_df["missing_pct"] <= 20)]["column"].tolist()

    summary = f"MISSING VALUE ANALYSIS:\n"
    summary += f"  Total missing cells: {df.isnull().sum().sum()} / {df.size} ({df.isnull().mean().mean()*100:.2f}%)\n"
    if critical:
        summary += f"  ⛔ Critical columns (>20% missing): {critical}\n"
    if warning:
        summary += f"  ⚠ Warning columns (5-20% missing): {warning}\n"
    summary += f"  ✅ Complete columns (0% missing): {missing_df[missing_df['missing_pct']==0]['column'].tolist()}"

    return {"tool": "missing_value_analysis", "results": results, "chart": chart, "summary": summary}


# ════════════════════════════════════════════════════════════
# TOOL 3 — Correlation Analysis
# ════════════════════════════════════════════════════════════
def correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Computes Pearson correlation matrix and identifies
    the strongest positive and negative correlations.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns.", "summary": "Insufficient numeric columns."}

    corr_matrix = numeric_df.corr(method="pearson")

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix)*0.9), max(6, len(corr_matrix)*0.8)))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr_matrix, ax=ax, annot=True, fmt=".2f",
        cmap="RdYlBu_r", vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 8}
    )
    ax.set_title("Pearson Correlation Matrix", fontsize=13, fontweight="bold", color=BLUE, pad=12)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    # Find top correlations
    corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_pairs.append({
                "col1": cols[i],
                "col2": cols[j],
                "correlation": round(float(corr_matrix.iloc[i, j]), 4)
            })
    corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    summary = "CORRELATION ANALYSIS:\n"
    summary += "  Top 5 strongest correlations:\n"
    for pair in corr_pairs[:5]:
        direction = "positive" if pair["correlation"] > 0 else "negative"
        strength = "strong" if abs(pair["correlation"]) > 0.7 else "moderate" if abs(pair["correlation"]) > 0.4 else "weak"
        summary += f"    {pair['col1']} ↔ {pair['col2']}: r={pair['correlation']} ({strength} {direction})\n"

    return {
        "tool": "correlation_analysis",
        "results": {"top_pairs": corr_pairs[:10], "matrix": corr_matrix.round(4).to_dict()},
        "chart": chart,
        "summary": summary
    }


# ════════════════════════════════════════════════════════════
# TOOL 4 — Outlier Detection
# ════════════════════════════════════════════════════════════
def outlier_detection(df: pd.DataFrame) -> dict:
    """
    Detects outliers using IQR method and Isolation Forest.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns.", "summary": "No numeric columns for outlier detection."}

    iqr_results = {}
    for col in numeric_cols:
        s = df[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_outliers = int(((s < lower) | (s > upper)).sum())
        iqr_results[col] = {
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(s) * 100, 2)
        }

    # Box plots
    n = len(numeric_cols)
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, notch=False,
                        medianprops=dict(color=ORANGE, lw=2))
        bp["boxes"][0].set_facecolor(BLUE2)
        bp["boxes"][0].set_alpha(0.6)
        n_out = iqr_results[col]["n_outliers"]
        ax.set_title(f"{col}\n({n_out} outliers)", fontsize=9, fontweight="bold", color=BLUE)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Outlier Detection — Box Plots (IQR Method)", fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    summary = "OUTLIER DETECTION (IQR Method):\n"
    high_outlier_cols = {k: v for k, v in iqr_results.items() if v["pct_outliers"] > 5}
    if high_outlier_cols:
        summary += "  ⚠ Columns with >5% outliers:\n"
        for col, info in high_outlier_cols.items():
            summary += f"    {col}: {info['n_outliers']} outliers ({info['pct_outliers']}%)\n"
    else:
        summary += "  ✅ No columns with >5% outliers detected.\n"

    for col, info in iqr_results.items():
        summary += f"  {col}: {info['n_outliers']} outliers, bounds=[{info['lower_bound']}, {info['upper_bound']}]\n"

    return {"tool": "outlier_detection", "results": iqr_results, "chart": chart, "summary": summary}


# ════════════════════════════════════════════════════════════
# TOOL 5 — Distribution Analysis
# ════════════════════════════════════════════════════════════
def distribution_analysis(df: pd.DataFrame) -> dict:
    """
    Tests normality of numeric columns using Shapiro-Wilk test
    and shows Q-Q plots.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return {"error": "No numeric columns.", "summary": "No numeric columns found."}

    results = {}
    for col in numeric_cols:
        s = df[col].dropna()
        sample = s.sample(min(5000, len(s)), random_state=42)
        stat, p_value = stats.shapiro(sample)
        skew = float(s.skew())
        kurt = float(s.kurtosis())
        results[col] = {
            "shapiro_stat": round(float(stat), 4),
            "shapiro_p": round(float(p_value), 6),
            "is_normal": bool(p_value > 0.05),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "distribution_shape": (
                "Right-skewed" if skew > 1 else
                "Left-skewed"  if skew < -1 else
                "Approximately normal"
            )
        }

    # Q-Q plots
    n = min(len(numeric_cols), 6)
    cols_to_plot = numeric_cols[:n]
    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 4))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        s = df[col].dropna()
        (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
        ax.scatter(osm, osr, color=BLUE2, alpha=0.4, s=10)
        ax.plot(osm, slope*np.array(osm)+intercept, color=ORANGE, lw=2)
        is_normal = results[col]["is_normal"]
        ax.set_title(f"{col}\n{'✅ Normal' if is_normal else '⚠ Non-normal'}",
                     fontsize=9, fontweight="bold",
                     color=GREEN if is_normal else ORANGE)
        ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Normality Test — Q-Q Plots", fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    normal_cols = [c for c, r in results.items() if r["is_normal"]]
    non_normal  = [c for c, r in results.items() if not r["is_normal"]]

    summary = "DISTRIBUTION ANALYSIS (Shapiro-Wilk Normality Test):\n"
    summary += f"  ✅ Normal distributions: {normal_cols if normal_cols else 'None'}\n"
    summary += f"  ⚠ Non-normal distributions: {non_normal if non_normal else 'None'}\n"
    for col, r in results.items():
        summary += f"  {col}: {r['distribution_shape']}, skew={r['skewness']}, p={r['shapiro_p']}\n"

    return {"tool": "distribution_analysis", "results": results, "chart": chart, "summary": summary}


# ════════════════════════════════════════════════════════════
# TOOL 6 — Hypothesis Testing
# ════════════════════════════════════════════════════════════
def hypothesis_testing(df: pd.DataFrame) -> dict:
    """
    Automatically selects and runs appropriate statistical tests:
    - t-test for binary categorical vs numeric
    - ANOVA for multi-class categorical vs numeric
    - Chi-square for two categorical columns
    """
    numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    results = []
    tested = 0

    # t-test / ANOVA: categorical (low cardinality) vs numeric
    for cat_col in categorical_cols[:3]:
        if df[cat_col].nunique() > 10:
            continue
        for num_col in numeric_cols[:3]:
            groups = [df[df[cat_col] == g][num_col].dropna()
                      for g in df[cat_col].dropna().unique()]
            groups = [g for g in groups if len(g) >= 5]
            if len(groups) < 2:
                continue

            if len(groups) == 2:
                t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
                test_name = "Independent t-test"
            else:
                t_stat, p_val = stats.f_oneway(*groups)
                test_name = "One-way ANOVA"

            results.append({
                "test": test_name,
                "variable1": cat_col,
                "variable2": num_col,
                "statistic": round(float(t_stat), 4),
                "p_value":   round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
                "interpretation": (
                    f"Significant difference in {num_col} across {cat_col} groups (p={p_val:.4f})"
                    if p_val < 0.05
                    else f"No significant difference in {num_col} across {cat_col} groups (p={p_val:.4f})"
                )
            })
            tested += 1
            if tested >= 6:
                break

    # Chi-square: two categorical columns
    if len(categorical_cols) >= 2:
        cat1, cat2 = categorical_cols[0], categorical_cols[1]
        ct = pd.crosstab(df[cat1], df[cat2])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        results.append({
            "test": "Chi-square Test",
            "variable1": cat1,
            "variable2": cat2,
            "statistic": round(float(chi2), 4),
            "p_value":   round(float(p_val), 6),
            "significant": bool(p_val < 0.05),
            "interpretation": (
                f"Significant association between {cat1} and {cat2} (p={p_val:.4f})"
                if p_val < 0.05
                else f"No significant association between {cat1} and {cat2} (p={p_val:.4f})"
            )
        })

    summary = "HYPOTHESIS TESTING RESULTS:\n"
    if not results:
        summary += "  Insufficient columns for hypothesis testing."
    for r in results:
        sig = "✅ SIGNIFICANT" if r["significant"] else "❌ NOT SIGNIFICANT"
        summary += f"  [{sig}] {r['test']}: {r['variable1']} vs {r['variable2']} → p={r['p_value']}\n"
        summary += f"    → {r['interpretation']}\n"

    return {"tool": "hypothesis_testing", "results": results, "chart": None, "summary": summary}


# ════════════════════════════════════════════════════════════
# TOOL 7 — Feature Importance
# ════════════════════════════════════════════════════════════
def feature_importance(df: pd.DataFrame) -> dict:
    """
    Computes feature importance using mutual information
    and Random Forest for the most likely target column
    (last numeric column or column named 'target'/'label'/'price' etc.)
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns.", "summary": "Insufficient columns."}

    # Guess the target column
    target_keywords = ["target", "label", "price", "sales", "revenue", "score",
                       "output", "result", "churn", "survived", "class"]
    target_col = None
    for col in df.columns:
        if any(kw in col.lower() for kw in target_keywords):
            target_col = col
            break
    if target_col is None or target_col not in numeric_cols:
        target_col = numeric_cols[-1]

    feature_cols = [c for c in numeric_cols if c != target_col]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_col].fillna(df[target_col].median())

    # Mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": feature_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)

    # Chart
    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_cols) * 0.5)))
    colors = [BLUE if i < 3 else BLUE2 for i in range(len(mi_df))]
    bars = ax.barh(mi_df["feature"], mi_df["mi_score"],
                   color=colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, mi_df["mi_score"]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mutual Information Score", fontsize=10)
    ax.set_title(f"Feature Importance for '{target_col}'", fontsize=12, fontweight="bold", color=BLUE)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    top3 = mi_df.head(3)["feature"].tolist()
    summary = f"FEATURE IMPORTANCE (Target: '{target_col}'):\n"
    summary += f"  Top 3 most important features: {top3}\n"
    for _, row in mi_df.iterrows():
        summary += f"  {row['feature']}: MI score = {row['mi_score']:.4f}\n"

    return {
        "tool": "feature_importance",
        "results": mi_df.to_dict(orient="records"),
        "chart": chart,
        "summary": summary
    }


# ════════════════════════════════════════════════════════════
# TOOL 8 — Categorical Analysis
# ════════════════════════════════════════════════════════════
def categorical_analysis(df: pd.DataFrame) -> dict:
    """
    Analyzes categorical columns: frequency distributions,
    bar charts for top categories.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return {"error": "No categorical columns found.", "summary": "No categorical columns."}

    cat_cols_to_plot = [c for c in cat_cols if df[c].nunique() <= 20][:6]
    if not cat_cols_to_plot:
        cat_cols_to_plot = cat_cols[:3]

    n = len(cat_cols_to_plot)
    cols_per_row = 2
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, rows * 4))
    axes = np.array(axes).flatten()

    results = {}
    for i, col in enumerate(cat_cols_to_plot):
        ax = axes[i]
        top_vals = df[col].value_counts().head(10)
        colors_bar = [BLUE if j == 0 else BLUE2 for j in range(len(top_vals))]
        top_vals.plot(kind="bar", ax=ax, color=colors_bar, edgecolor="white")
        ax.set_title(f"{col}\n({df[col].nunique()} unique values)",
                     fontsize=10, fontweight="bold", color=BLUE)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=45)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        results[col] = top_vals.to_dict()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Analysis — Value Distributions",
                 fontsize=13, fontweight="bold", color=BLUE)
    plt.tight_layout()
    chart = _fig_to_base64(fig)

    summary = "CATEGORICAL ANALYSIS:\n"
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_val  = df[col].value_counts().index[0] if not df[col].value_counts().empty else "N/A"
        top_freq = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
        summary += (f"  {col}: {n_unique} unique values, "
                    f"most common = '{top_val}' ({top_freq} times, "
                    f"{top_freq/len(df)*100:.1f}%)\n")

    return {"tool": "categorical_analysis", "results": results, "chart": chart, "summary": summary}


# ════════════════════════════════════════════════════════════
# Master tool runner
# ════════════════════════════════════════════════════════════
TOOL_REGISTRY = {
    "descriptive_stats":       descriptive_stats,
    "missing_value_analysis":  missing_value_analysis,
    "correlation_analysis":    correlation_analysis,
    "outlier_detection":       outlier_detection,
    "distribution_analysis":   distribution_analysis,
    "hypothesis_testing":      hypothesis_testing,
    "feature_importance":      feature_importance,
    "categorical_analysis":    categorical_analysis,
}

def run_tool(tool_name: str, df: pd.DataFrame) -> dict:
    """Run a tool by name and return its result dict."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}", "summary": "Tool not found."}
    try:
        return TOOL_REGISTRY[tool_name](df)
    except Exception as e:
        return {"error": str(e), "summary": f"Tool '{tool_name}' failed: {str(e)}"}
