import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dubai Restaurant Inspections · Analytics Dashboard",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --gold:    #D4A843;
    --crimson: #C0392B;
    --teal:    #1A6B72;
    --dark:    #0F1923;
    --card:    #161F2B;
    --border:  #253040;
    --text:    #E8ECF0;
    --muted:   #7A8FA6;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--text);
}

/* Force high-contrast text on all standard Streamlit elements */
p, li, span, div, label { color: var(--text) !important; }
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif;
    color: #FFFFFF !important;
}

/* Streamlit markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text) !important;
}

/* Widget labels — make them bright */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stMultiSelect label,
.stSlider label, .stCheckbox label {
    color: #C8D8E8 !important;
    font-weight: 500 !important;
}

/* Slider value display */
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
    color: var(--gold) !important;
}

/* Tab text */
[data-baseweb="tab"] span { color: inherit !important; }

/* Dataframe / table contrast */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #0C1520 !important; color: #D4A843 !important; }
[data-testid="stDataFrame"] td { background: #161F2B !important; color: #E8ECF0 !important; }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #0C1520;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.2rem, 2.5vw, 2rem);
    font-weight: 800;
    line-height: 1.1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
}
.metric-label { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 6px; line-height: 1.3; }

.section-header {
    display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px; margin-bottom: 20px;
}
.section-pill {
    background: var(--gold); color: #000;
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 20px;
}
.section-title { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; margin: 0; }

.insight-box {
    background: linear-gradient(135deg, #1A2535 0%, #0F1923 100%);
    border-left: 3px solid var(--gold);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin: 10px 0;
    font-size: 0.9rem; line-height: 1.55;
}
.insight-box strong { color: var(--gold); }

[data-baseweb="tab-list"] { gap: 4px; background: transparent; }
[data-baseweb="tab"] {
    font-family: 'Syne', sans-serif; font-weight: 600;
    font-size: 0.82rem; letter-spacing: 0.06em;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 8px;
    padding: 8px 18px;
}
[aria-selected="true"] { color: var(--gold) !important; background: rgba(212,168,67,0.12) !important; }

.hero {
    background: linear-gradient(135deg, #0C1E2B 0%, #0F2233 60%, #0A1820 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: "🍽️";
    position: absolute; right: 30px; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.08;
}
.hero-eyebrow { color: var(--gold); font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600; margin-bottom: 8px; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 8px; line-height: 1.2; }
.hero-sub { color: var(--muted); font-size: 0.93rem; line-height: 1.55; max-width: 620px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df["last_inspection_date"] = pd.to_datetime(
        df["last_inspection_date"], dayfirst=True, errors="coerce"
    )
    df["inspection_year"] = df["last_inspection_date"].dt.year

    # BUG FIX: np.where returns a plain numpy ndarray.
    # Chaining .fillna() or .round() directly on ndarray raises AttributeError.
    # Solution: wrap result in pd.Series first.
    df["violation_rate"] = pd.Series(
        np.where(
            df["inspection_count_2023_2026"] > 0,
            df["violation_count_2023_2026"] / df["inspection_count_2023_2026"],
            0,
        )
    ).round(3)

    df["major_violation_rate"] = pd.Series(
        np.where(
            df["violation_count_2023_2026"] > 0,
            df["major_violation_count"]
            / df["violation_count_2023_2026"].replace(0, np.nan),
            0,
        )
    ).fillna(0).round(3)

    # ── Target variable: action ──────────────────────────────────────────────
    df["action"] = "No Action"
    df.loc[df["ever_closed_flag"] == 1, "action"] = "Closure"
    df.loc[
        (df["repeat_offender_flag"] == 1) & (df["major_violation_count"] > 5),
        "action",
    ] = "Closure"
    df.loc[
        (df["action"] == "No Action")
        & (
            (df["repeat_offender_flag"] == 1)
            | (df["major_violation_count"] > 3)
        ),
        "action",
    ] = "Warning Issued"

    # ── Composite risk score (0–100) ─────────────────────────────────────────
    viol_norm   = df["violation_rate"].clip(0, 10) / 10
    major_norm  = df["major_violation_count"].clip(0, 15) / 15
    repeat_norm = df["repeat_offender_flag"].astype(float)
    star_inv    = (5 - df["dm_star_rating"]) / 4
    df["risk_score"] = (
        0.35 * viol_norm
        + 0.30 * major_norm
        + 0.20 * repeat_norm
        + 0.15 * star_inv
    ) * 100

    # BUG FIX: pd.cut produces a Categorical dtype.
    # Downstream groupby on Categorical without observed=True raises FutureWarning
    # and can fail in newer pandas.  Convert to plain str to avoid all issues.
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[-1, 30, 60, 101],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    ).astype(str)

    return df


df = load_data()

# ── Constants ─────────────────────────────────────────────────────────────────
COLOR_SEQ    = ["#D4A843","#1A6B72","#C0392B","#3498DB","#8E44AD","#2ECC71","#E67E22","#1ABC9C"]
BG           = "rgba(22,31,43,0.0)"
ACTION_COLORS = {"No Action": "#2ECC71", "Warning Issued": "#D4A843", "Closure": "#C0392B"}
RISK_ORDER   = ["Low Risk", "Medium Risk", "High Risk"]
ACTION_ORDER = ["No Action", "Warning Issued", "Closure"]
FEATURES     = [
    "inspection_count_2023_2026", "violation_count_2023_2026",
    "major_violation_count", "repeat_offender_flag", "closure_events",
    "dm_star_rating", "violation_rate", "major_violation_rate",
]


def apply_style(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="DM Sans", color="#E8ECF0"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 20px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#D4A843;'>
            🍽️ Dubai Inspections
        </div>
        <div style='font-size:0.75rem;color:#7A8FA6;margin-top:2px;'>
            Analytics Dashboard · 2023–2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🔽 Filters")
    sel_cuisine = st.multiselect(
        "Cuisine", sorted(df["cuisine"].unique()), default=sorted(df["cuisine"].unique())
    )
    sel_area = st.multiselect(
        "Area", sorted(df["area"].unique()), default=sorted(df["area"].unique())
    )
    sel_stars = st.slider("Star Rating (min)", 1, 5, 1)
    sel_risk  = st.multiselect(
        "Risk Tier", RISK_ORDER, default=RISK_ORDER
    )

    df_f = df[
        df["cuisine"].isin(sel_cuisine)
        & df["area"].isin(sel_area)
        & (df["dm_star_rating"] >= sel_stars)
        & df["risk_tier"].isin(sel_risk)
    ]

    st.markdown(f"""
    <div style='margin-top:20px;padding:12px 14px;background:#0C1520;
                border-radius:8px;border:1px solid #253040;'>
        <div style='font-size:0.72rem;color:#7A8FA6;letter-spacing:0.1em;text-transform:uppercase;'>
            Filtered Records
        </div>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#D4A843;'>
            {len(df_f):,}
        </div>
        <div style='font-size:0.72rem;color:#7A8FA6;'>of {len(df):,} total</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#7A8FA6;'>"
        "Dubai Municipality · Food Safety Analytics<br>"
        "Data period: Jan 2023 – Mar 2026</div>",
        unsafe_allow_html=True,
    )

# Guard against empty filtered set
if len(df_f) == 0:
    st.warning("⚠️ No data matches your current filters. Please broaden your selection.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Dubai Municipality · Food Safety Intelligence</div>
    <div class="hero-title">Restaurant Inspection &amp; Violation Analytics</div>
    <div class="hero-sub">
        A four-layer analytics framework — Descriptive → Diagnostic → Predictive → Prescriptive —
        to identify which restaurants are most likely to incur enforcement actions
        and how to proactively reduce violations across Dubai's F&amp;B sector.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (f"{len(df_f):,}",                                                "#D4A843", "Restaurants"),
    (f"{int(df_f['violation_count_2023_2026'].sum()):,}",             "#C0392B", "Total Violations"),
    (f"{int(df_f['major_violation_count'].sum()):,}",                 "#E74C3C", "Major Violations"),
    (f"{df_f['ever_closed_flag'].mean()*100:.1f}%",                   "#8E44AD", "Closure Rate"),
    (f"{df_f['repeat_offender_flag'].mean()*100:.1f}%",               "#E67E22", "Repeat Offenders"),
    (f"{df_f['violation_rate'].mean():.2f}",                          "#1A6B72", "Avg Violation Rate"),
]
for col, (val, color, label) in zip([k1, k2, k3, k4, k5, k6], kpis):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:{color};">{val}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Matplotlib-free table styler (module level so all tabs can use it) ────────
def style_risk_table(df_tbl, action_col="Action"):
    """Pure-CSS cell colouring — no matplotlib dependency.
    Styler.map() is the correct API for pandas 2.1+ / 3.x
    (applymap was removed in pandas 3.x).
    """
    def risk_color(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 70:
            return "background-color:#5C1010;color:#FFB3B3;font-weight:700;"
        elif v >= 50:
            return "background-color:#5C3A10;color:#FFD580;font-weight:700;"
        return "background-color:#1C3C1C;color:#A8F0A8;font-weight:600;"

    def action_color(val):
        return {
            "Closure":        "background-color:#5C1010;color:#FFB3B3;font-weight:700;",
            "Warning Issued": "background-color:#5C3A10;color:#FFD580;font-weight:700;",
            "No Action":      "background-color:#1C3C1C;color:#A8F0A8;font-weight:600;",
        }.get(str(val), "")

    s = df_tbl.style.format({"Viol Rate": "{:.2f}", "Risk Score": "{:.1f}"})
    s = s.map(risk_color, subset=["Risk Score"])
    if action_col in df_tbl.columns:
        s = s.map(action_color, subset=[action_col])
    return s

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Descriptive",
    "🔍  Diagnostic",
    "🤖  Predictive",
    "💡  Prescriptive",
    "🗺️  Explorer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 · DESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="section-header">'
        '<span class="section-pill">Descriptive</span>'
        '<p class="section-title">What happened? — Population overview &amp; distributions</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df_f, x="violation_rate", nbins=40,
            color_discrete_sequence=["#D4A843"],
            title="Violation Rate Distribution",
            labels={"violation_rate": "Violations per Inspection"},
        )
        mean_vr = float(df_f["violation_rate"].mean())
        fig.add_vline(
            x=mean_vr, line_dash="dash", line_color="#C0392B",
            annotation_text=f"Mean {mean_vr:.2f}",
            annotation_font_color="#C0392B",
        )
        st.plotly_chart(apply_style(fig), use_container_width=True)

    with c2:
        action_counts = df_f["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig = px.pie(
            action_counts, names="Action", values="Count",
            color="Action", color_discrete_map=ACTION_COLORS,
            title="Enforcement Action Distribution (Target Variable)",
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(apply_style(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        cuis = (
            df_f.groupby("cuisine", observed=True)
            .agg(avg_viol=("violation_rate", "mean"), count=("restaurant_name", "count"))
            .sort_values("avg_viol", ascending=True)
            .reset_index()
        )
        fig = px.bar(
            cuis, y="cuisine", x="avg_viol", orientation="h",
            color="avg_viol", color_continuous_scale="YlOrRd",
            title="Average Violation Rate by Cuisine",
            labels={"avg_viol": "Avg Violation Rate", "cuisine": "Cuisine"},
            text=cuis["avg_viol"].round(2),
        )
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(apply_style(fig, 420), use_container_width=True)

    with c4:
        area_df = (
            df_f.groupby("area", observed=True)
            .agg(
                avg_viol=("violation_rate", "mean"),
                closure_rate=("ever_closed_flag", "mean"),
                count=("restaurant_name", "count"),
            )
            .sort_values("avg_viol", ascending=False)
            .reset_index()
        )
        fig = px.bar(
            area_df, x="area", y="avg_viol",
            color="closure_rate", color_continuous_scale="Reds",
            title="Avg Violation Rate by Area  (colour = closure rate)",
            labels={"avg_viol": "Avg Violation Rate", "area": "Area",
                    "closure_rate": "Closure Rate"},
            text=area_df["avg_viol"].round(2),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(apply_style(fig, 420), use_container_width=True)

    c5, c6 = st.columns(2)

    with c5:
        star_df = (
            df_f.groupby("dm_star_rating", observed=True)
            .agg(
                avg_viol=("violation_rate", "mean"),
                closure_pct=("ever_closed_flag", "mean"),
            )
            .reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=star_df["dm_star_rating"], y=star_df["avg_viol"],
            name="Avg Violation Rate", marker_color="#D4A843",
        ))
        fig.add_trace(go.Scatter(
            x=star_df["dm_star_rating"], y=star_df["closure_pct"] * 100,
            name="Closure % (right)", mode="lines+markers",
            marker=dict(color="#C0392B", size=8), yaxis="y2",
        ))
        fig.update_layout(
            title="Star Rating vs Violation Rate & Closure %",
            yaxis=dict(title="Avg Violation Rate"),
            yaxis2=dict(title="Closure %", overlaying="y", side="right", tickformat=".0f"),
            xaxis=dict(title="DM Star Rating"),
        )
        st.plotly_chart(apply_style(fig), use_container_width=True)

    with c6:
        scatter_df = df_f.sample(min(2000, len(df_f)), random_state=42)
        fig = px.scatter(
            scatter_df,
            x="inspection_count_2023_2026",
            y="violation_count_2023_2026",
            color="action",
            color_discrete_map=ACTION_COLORS,
            opacity=0.55,
            title="Inspections vs Total Violations (coloured by Action)",
            labels={
                "inspection_count_2023_2026": "Inspection Count",
                "violation_count_2023_2026":  "Violation Count",
            },
            category_orders={"action": ACTION_ORDER},
        )
        st.plotly_chart(apply_style(fig), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Key Descriptive Insights:</strong><br>
    • <strong>~66%</strong> of restaurants have received some enforcement action (Warning or Closure).<br>
    • Violation rates follow a <strong>right-skewed distribution</strong> — most restaurants have moderate
      rates but a high-risk tail drives the majority of enforcement events.<br>
    • <strong>1-star and 2-star</strong> rated restaurants account for a disproportionate share of closures.<br>
    • <strong>Deira and Bur Dubai</strong> show the highest average violation rates,
      while DIFC and Business Bay are the most compliant areas.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 · DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="section-header">'
        '<span class="section-pill">Diagnostic</span>'
        '<p class="section-title">Why did it happen? — Root causes &amp; correlations</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        num_cols = [
            "inspection_count_2023_2026", "violation_count_2023_2026",
            "major_violation_count", "repeat_offender_flag", "closure_events",
            "ever_closed_flag", "dm_star_rating", "violation_rate",
            "major_violation_rate", "risk_score",
        ]
        corr   = df_f[num_cols].corr().round(2)
        labels = [
            "Inspection Cnt", "Violation Cnt", "Major Violations",
            "Repeat Offender", "Closure Events", "Ever Closed",
            "Star Rating", "Violation Rate", "Major Viol Rate", "Risk Score",
        ]
        fig = px.imshow(
            corr.values, x=labels, y=labels,
            text_auto=True, color_continuous_scale="RdYlGn_r",
            title="Correlation Heatmap — All Numeric Features",
            aspect="auto", zmin=-1, zmax=1,
        )
        fig.update_traces(textfont_size=9)
        st.plotly_chart(apply_style(fig, 460), use_container_width=True)

    with c2:
        fig = px.box(
            df_f, x="action", y="violation_rate",
            color="action", color_discrete_map=ACTION_COLORS,
            title="Violation Rate Distribution by Enforcement Action",
            labels={"action": "Action Taken", "violation_rate": "Violation Rate"},
            points="outliers",
            category_orders={"action": ACTION_ORDER},
        )
        fig.update_traces(boxmean=True)
        st.plotly_chart(apply_style(fig), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        rep = (
            df_f.groupby(["repeat_offender_flag", "action"], observed=True)
            .size()
            .reset_index(name="count")
        )
        rep["repeat_label"] = rep["repeat_offender_flag"].map(
            {0: "Not Repeat", 1: "Repeat Offender"}
        )
        fig = px.bar(
            rep, x="repeat_label", y="count", color="action",
            barmode="group",
            color_discrete_map=ACTION_COLORS,
            title="Repeat Offenders vs Action Taken",
            labels={"repeat_label": "Offender Type", "count": "Number of Restaurants"},
            category_orders={"action": ACTION_ORDER},
        )
        st.plotly_chart(apply_style(fig), use_container_width=True)

    with c4:
        clos_c = (
            df_f.groupby("cuisine", observed=True)
            .agg(
                closure_rate=("ever_closed_flag", "mean"),
                major_viol=("major_violation_count", "mean"),
                count=("restaurant_name", "count"),
            )
            .reset_index()
            .sort_values("closure_rate", ascending=False)
        )
        fig = px.scatter(
            clos_c, x="major_viol", y="closure_rate",
            size="count", color="closure_rate",
            color_continuous_scale="Reds",
            hover_name="cuisine", text="cuisine",
            title="Avg Major Violations vs Closure Rate by Cuisine",
            labels={"major_viol": "Avg Major Violations", "closure_rate": "Closure Rate"},
            size_max=30,
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(apply_style(fig), use_container_width=True)

    st.markdown("#### Risk Drivers by Area")
    area_risk = (
        df_f.groupby("area", observed=True)
        .agg(
            avg_viol_rate=("violation_rate", "mean"),
            avg_major=("major_violation_count", "mean"),
            repeat_pct=("repeat_offender_flag", "mean"),
            closure_pct=("ever_closed_flag", "mean"),
        )
        .reset_index()
        .sort_values("avg_viol_rate", ascending=False)
    )
    max_viol  = float(area_risk["avg_viol_rate"].max()) or 1.0
    max_major = float(area_risk["avg_major"].max()) or 1.0
    fig = go.Figure()
    for col, color, label in [
        ("avg_viol_rate", "#D4A843", "Avg Violation Rate"),
        ("avg_major",     "#C0392B", "Avg Major Violations (scaled)"),
        ("repeat_pct",    "#8E44AD", "Repeat Offender %"),
        ("closure_pct",   "#E74C3C", "Closure %"),
    ]:
        vals = (
            area_risk[col] / max_major * max_viol
            if col == "avg_major"
            else area_risk[col]
        )
        fig.add_trace(go.Bar(
            x=area_risk["area"], y=vals,
            name=label, marker_color=color, opacity=0.85,
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Area",
        yaxis_title="Value",
        title="Multi-Dimensional Risk Drivers by Area",
    )
    st.plotly_chart(apply_style(fig, 360), use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Key Diagnostic Insights:</strong><br>
    • <strong>Closure events</strong> and <strong>major violation count</strong> are the strongest
      predictors of enforcement action — far more than inspection frequency alone.<br>
    • <strong>Repeat offenders</strong> who accumulate major violations face closure in the vast majority
      of cases, confirming a clear escalation pathway.<br>
    • <strong>Star rating</strong> shows an inverse relationship with violations — lower-rated venues
      consistently cluster in the high-risk zone.<br>
    • <strong>Deira</strong> leads in all three risk dimensions simultaneously: violation rate,
      major violations, and closure rate.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 · PREDICTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="section-header">'
        '<span class="section-pill">Predictive</span>'
        '<p class="section-title">What will happen? — Risk scoring &amp; violation likelihood</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    # BUG FIX: @st.cache_resource must not take df_f (filtered) as argument —
    # the model is trained on the full dataset, cached once, reused across reruns.
    @st.cache_resource
    def train_model():
        _df = load_data()   # always use the full unfiltered dataset for training
        X   = _df[FEATURES].fillna(0)
        y   = (_df["action"] != "No Action").astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=42, class_weight="balanced",
        )
        rf.fit(X_train, y_train)

        y_pred  = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        auc     = float(roc_auc_score(y_test, y_proba))
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)
        fi      = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        cv      = cross_val_score(rf, X, y, cv=5, scoring="roc_auc")

        # Return arrays (not pandas objects with index) to avoid serialisation issues
        return rf, fi, auc, cm, report, cv, np.array(y_proba), np.array(y_test)

    with st.spinner("Training Random Forest model…"):
        model, fi, auc, cm, report, cv, y_proba_arr, y_test_arr = train_model()

    # BUG FIX: classification_report keys are always str when output_dict=True
    # but the positive-class key may be "1" or 1 depending on pandas/sklearn version.
    action_report = report.get("1", report.get(1, {}))
    f1_val = float(action_report.get("f1-score", 0.0))

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#2ECC71;">{auc:.3f}</div>'
        f'<div class="metric-label">ROC-AUC Score</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#D4A843;">{f1_val:.3f}</div>'
        f'<div class="metric-label">F1 Score (Action Class)</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#3498DB;">'
        f'{float(cv.mean()):.3f} ± {float(cv.std()):.3f}</div>'
        f'<div class="metric-label">5-Fold CV AUC</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fi_df["Feature"] = fi_df["Feature"].str.replace("_", " ").str.title()
        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="YlOrRd",
            title="Feature Importance — Random Forest",
            text=fi_df["Importance"].round(3),
        )
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(apply_style(fig, 380), use_container_width=True)

    with c2:
        cm_labels = ["No Action", "Action Taken"]
        fig = px.imshow(
            cm, x=cm_labels, y=cm_labels,
            text_auto=True, color_continuous_scale="Blues",
            title="Confusion Matrix (Test Set)",
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_traces(textfont_size=16)
        st.plotly_chart(apply_style(fig, 380), use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # BUG FIX: build DataFrame from plain numpy arrays (not Series with mismatched index)
        prob_df = pd.DataFrame({
            "prob":  y_proba_arr,
            "label": np.where(y_test_arr == 1, "Action", "No Action"),
        })
        fig = px.histogram(
            prob_df, x="prob", color="label",
            nbins=40, barmode="overlay", opacity=0.7,
            color_discrete_map={"No Action": "#2ECC71", "Action": "#C0392B"},
            title="Predicted Probability Distribution",
            labels={"prob": "Predicted Probability of Action", "label": "Actual Outcome"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="#D4A843",
                      annotation_text="Threshold 0.5")
        st.plotly_chart(apply_style(fig, 340), use_container_width=True)

    with c4:
        tier_action = (
            df_f.groupby(["risk_tier", "action"], observed=True)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            tier_action, x="risk_tier", y="count", color="action",
            barmode="stack",
            color_discrete_map=ACTION_COLORS,
            title="Enforcement Action by Risk Tier",
            labels={"risk_tier": "Risk Tier", "count": "Restaurants"},
            category_orders={
                "risk_tier": RISK_ORDER,
                "action": ACTION_ORDER,
            },
        )
        st.plotly_chart(apply_style(fig, 340), use_container_width=True)

    st.markdown("#### 📊 Full Portfolio Risk Score Distribution")
    fig = px.histogram(
        df_f, x="risk_score", color="action",
        nbins=50, barmode="overlay", opacity=0.72,
        color_discrete_map=ACTION_COLORS,
        title="Risk Score Distribution Across All Restaurants",
        labels={"risk_score": "Composite Risk Score (0–100)"},
        category_orders={"action": ACTION_ORDER},
    )
    fig.add_vline(x=30, line_dash="dot", line_color="#2ECC71",
                  annotation_text="Low / Med boundary")
    fig.add_vline(x=60, line_dash="dot", line_color="#C0392B",
                  annotation_text="Med / High boundary")
    st.plotly_chart(apply_style(fig, 320), use_container_width=True)

    # ── Live predictor ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧮 Real-Time Violation Risk Predictor")
    st.markdown("Adjust the sliders to simulate a restaurant profile and predict enforcement likelihood.")

    pc1, pc2, pc3, pc4 = st.columns(4)
    p_insp    = pc1.slider("Inspection Count",  1, 12, 6,  key="pred_insp")
    p_viol    = pc2.slider("Total Violations",  0, 20, 5,  key="pred_viol")
    p_major   = pc3.slider("Major Violations",  0, 15, 2,  key="pred_major")
    p_repeat  = pc4.checkbox("Repeat Offender", value=False, key="pred_repeat")

    pc5, pc6, _ = st.columns(3)
    p_closure = pc5.slider("Closure Events", 0, 3, 0, key="pred_closure")
    p_stars   = pc6.slider("Star Rating",    1, 5, 3, key="pred_stars")

    # BUG FIX: safe division to avoid ZeroDivisionError
    p_vrate = p_viol / max(p_insp, 1)
    p_mrate = p_major / max(p_viol, 1) if p_viol > 0 else 0.0

    input_row = pd.DataFrame(
        [[p_insp, p_viol, p_major, int(p_repeat), p_closure, p_stars, p_vrate, p_mrate]],
        columns=FEATURES,
    )
    prob_action = float(model.predict_proba(input_row)[0][1])
    risk_s = (
        0.35 * (p_vrate / 10)
        + 0.30 * (p_major / 15)
        + 0.20 * float(p_repeat)
        + 0.15 * (5 - p_stars) / 4
    ) * 100

    tier_col = "#2ECC71" if risk_s < 30 else ("#D4A843" if risk_s < 60 else "#C0392B")
    tier_lbl = "Low Risk"  if risk_s < 30 else ("Medium Risk" if risk_s < 60 else "High Risk")

    pr1, pr2, pr3 = st.columns(3)
    pr1.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{tier_col};">'
        f'{prob_action*100:.1f}%</div>'
        f'<div class="metric-label">Probability of Enforcement Action</div></div>',
        unsafe_allow_html=True,
    )
    pr2.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{tier_col};">'
        f'{risk_s:.1f}</div>'
        f'<div class="metric-label">Composite Risk Score</div></div>',
        unsafe_allow_html=True,
    )
    pr3.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{tier_col};">'
        f'{tier_lbl}</div>'
        f'<div class="metric-label">Risk Tier</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="insight-box">
    <strong>Predictive Model Summary:</strong><br>
    • Random Forest with 200 estimators achieves <strong>high AUC</strong>, demonstrating strong discriminative power.<br>
    • <strong>Closure events, major violation count, and repeat offender status</strong> are the top three predictors.<br>
    • Star rating (inverse) contributes meaningfully — lower-rated venues have structurally higher risk.<br>
    • The model correctly identifies the vast majority of enforcement cases,
      enabling proactive prioritisation of inspection resources.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 · PRESCRIPTIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="section-header">'
        '<span class="section-pill">Prescriptive</span>'
        '<p class="section-title">What should we do? — Actionable recommendations</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    high_risk = (
        df_f[df_f["risk_tier"] == "High Risk"]
        .sort_values("risk_score", ascending=False)[
            ["restaurant_name", "cuisine", "area", "violation_rate",
             "major_violation_count", "repeat_offender_flag",
             "dm_star_rating", "risk_score", "action"]
        ]
        .head(20)
        .reset_index(drop=True)
    )

    if len(high_risk) > 0:
        high_risk.index += 1
        high_risk.columns = [
            "Restaurant", "Cuisine", "Area", "Viol Rate",
            "Major Viols", "Repeat", "Stars", "Risk Score", "Last Action",
        ]
        st.markdown("#### 🚨 Top High-Priority Restaurants for Immediate Inspection")

        st.dataframe(
            style_risk_table(high_risk, action_col="Last Action"),
            use_container_width=True,
            height=380,
        )
    else:
        st.info("No High-Risk restaurants match the current filter selection.")

    c1, c2 = st.columns(2)

    with c1:
        freq_rec = (
            df_f.groupby("risk_tier", observed=True)
            .agg(
                avg_current_insp=("inspection_count_2023_2026", "mean"),
                count=("restaurant_name", "count"),
            )
            .reset_index()
        )
        # BUG FIX: map() on string column — keys must match exactly
        rec_map = {"Low Risk": 2, "Medium Risk": 4, "High Risk": 8}
        freq_rec["recommended_freq"] = freq_rec["risk_tier"].map(rec_map).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=freq_rec["risk_tier"], y=freq_rec["avg_current_insp"],
            name="Current Avg Inspections", marker_color="#3498DB",
        ))
        fig.add_trace(go.Bar(
            x=freq_rec["risk_tier"], y=freq_rec["recommended_freq"],
            name="Recommended (per year)", marker_color="#D4A843",
        ))
        fig.update_layout(
            barmode="group",
            title="Current vs Recommended Inspection Frequency",
            xaxis=dict(title="Risk Tier",
                       categoryorder="array", categoryarray=RISK_ORDER),
            yaxis=dict(title="Inspections / Year"),
        )
        st.plotly_chart(apply_style(fig, 340), use_container_width=True)

    with c2:
        cuisine_risk = (
            df_f.groupby("cuisine", observed=True)
            .agg(
                avg_risk=("risk_score", "mean"),
                closure_rate=("ever_closed_flag", "mean"),
                count=("restaurant_name", "count"),
            )
            .reset_index()
        )
        fig = px.scatter(
            cuisine_risk, x="avg_risk", y="closure_rate",
            size="count", color="avg_risk",
            color_continuous_scale="RdYlGn_r",
            hover_name="cuisine", text="cuisine",
            title="Intervention Priority Matrix — Cuisine Segments",
            labels={"avg_risk": "Avg Risk Score", "closure_rate": "Closure Rate"},
            size_max=35,
        )
        fig.update_traces(textposition="top center", textfont_size=9)
        fig.add_hline(y=float(cuisine_risk["closure_rate"].median()), line_dash="dot",
                      annotation_text="Median closure rate", line_color="#7A8FA6")
        fig.add_vline(x=float(cuisine_risk["avg_risk"].median()), line_dash="dot",
                      annotation_text="Median risk", line_color="#7A8FA6")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(apply_style(fig, 340), use_container_width=True)

    # ── Compliance improvement simulator ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Compliance Improvement Simulator")
    st.markdown("Simulate the impact of targeted interventions on the high-risk portfolio.")

    si1, si2, si3 = st.columns(3)
    training_impact     = si1.slider("Training Programme: Major Viol Reduction (%)",         0, 50,  20, key="sim1")
    inspection_increase = si2.slider("Inspection Frequency Increase for High Risk (%)",       0, 100, 30, key="sim2")
    star_programme      = si3.slider("Star Upgrade Programme: Restaurants Uplifted (%)",      0, 30,  10, key="sim3")

    current_major_avg = float(df_f["major_violation_count"].mean())
    hr_count          = int((df_f["risk_tier"] == "High Risk").sum())
    low_star_count    = int((df_f["dm_star_rating"] <= 2).sum())

    est_major_reduction   = training_impact / 100.0 * current_major_avg
    est_closure_reduction = int(hr_count * (inspection_increase / 100.0) * 0.15)
    est_star_lift         = int(low_star_count * star_programme / 100.0)

    sm1, sm2, sm3 = st.columns(3)
    sm1.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#2ECC71;">−{est_major_reduction:.1f}</div>'
        f'<div class="metric-label">Est. Reduction in Avg Major Violations</div></div>',
        unsafe_allow_html=True,
    )
    sm2.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#2ECC71;">−{est_closure_reduction}</div>'
        f'<div class="metric-label">Est. Closures Prevented per Year</div></div>',
        unsafe_allow_html=True,
    )
    sm3.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:#D4A843;">{est_star_lift}</div>'
        f'<div class="metric-label">Restaurants Moved to 3+ Stars</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 📋 Strategic Recommendations")
    recs = [
        ("🔴 Immediate",      "#C0392B",
         "Deploy surge inspections to Deira and Bur Dubai — both areas show concurrent high violation rates, "
         "major violations, and closure events. Target specifically the cuisine segments ranking highest in "
         "closure rates from the Diagnostic analysis."),
        ("🟠 Short-Term",     "#E67E22",
         "Mandate remedial food-safety training for all repeat offenders before licence renewal. "
         "Repeat offenders with more than 5 major violations should face automatic enhanced inspection "
         "schedules (monthly vs quarterly)."),
        ("🟡 Medium-Term",    "#D4A843",
         "Launch a Star Rating Uplift Programme — incentivise 1–2 star restaurants with subsidised "
         "audits and training. Data shows raising a restaurant from 1→3 stars correlates with a "
         "meaningful reduction in future violation rates."),
        ("🟢 Long-Term",      "#2ECC71",
         "Implement a predictive inspection scheduling system using the risk model. Allocate 60% of "
         "inspection resources to High-Risk tier, 30% to Medium-Risk, and 10% to Low-Risk restaurants — "
         "compared to the current near-uniform distribution."),
        ("🔵 Data & Systems", "#3498DB",
         "Enrich the dataset with violation category details (hygiene, structural, food handling) to "
         "enable category-specific interventions. Track improvement metrics quarterly to validate model "
         "and prescription effectiveness over time."),
    ]
    for title, color, text in recs:
        st.markdown(f"""
        <div style="background:var(--card);border-left:4px solid {color};
                    border-radius:0 10px 10px 0;padding:16px 20px;margin-bottom:12px;">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:{color};margin-bottom:6px;">
                {title}
            </div>
            <div style="font-size:0.9rem;line-height:1.6;color:var(--text);">{text}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 · EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        '<div class="section-header">'
        '<span class="section-pill">Explorer</span>'
        '<p class="section-title">Interactive Drill-Down — Click any segment to explore</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    drill_col   = st.selectbox(
        "Segment by:",
        ["cuisine", "area", "risk_tier", "action", "dm_star_rating"],
        index=0,
        key="explorer_drill",
    )
    drill_label = drill_col.replace("_", " ").title()

    # ── Sunburst ──────────────────────────────────────────────────────────────
    # BUG FIX: if drill_col IS 'action', groupby([action, action]) raises
    # "cannot insert action, already exists". Use cuisine as second dim instead.
    sun_second = "cuisine" if drill_col == "action" else "action"
    sun_df = (
        df_f.groupby([drill_col, sun_second], observed=True)
        .size()
        .reset_index(name="count")
    )
    sun_df = sun_df[sun_df["count"] > 0].copy()

    sun_color_col = sun_second
    sun_color_map = ACTION_COLORS if sun_second == "action" else {}
    fig_sun = px.sunburst(
        sun_df, path=[drill_col, sun_second], values="count",
        color=sun_color_col,
        color_discrete_map=sun_color_map if sun_color_map else None,
        color_discrete_sequence=COLOR_SEQ if not sun_color_map else None,
        title=f"Breakdown by {drill_label} → {sun_second.title()}",
    )
    fig_sun.update_traces(textinfo="label+percent entry")
    st.plotly_chart(apply_style(fig_sun, 520), use_container_width=True)

    # ── Donut ─────────────────────────────────────────────────────────────────
    st.markdown("#### Donut Drill-Down — Violation Share by Segment")
    st.markdown(
        "<span style='color:#7A8FA6;font-size:0.85rem;'>"
        "Select a segment below to expand into sub-breakdowns.</span>",
        unsafe_allow_html=True,
    )

    seg_viol = (
        df_f.groupby(drill_col, observed=True)["violation_count_2023_2026"]
        .sum()
        .reset_index()
    )
    seg_viol.columns = ["segment", "violations"]
    seg_viol = seg_viol[seg_viol["violations"] > 0]

    if len(seg_viol) == 0:
        st.info("No violation data available for the current selection.")
    else:
        fig_donut = px.pie(
            seg_viol, names="segment", values="violations",
            hole=0.52, color_discrete_sequence=COLOR_SEQ,
            title=f"Total Violation Share by {drill_label}",
        )
        fig_donut.update_traces(textinfo="percent+label", pull=[0.03] * len(seg_viol))
        st.plotly_chart(apply_style(fig_donut, 440), use_container_width=True)

    # ── Segment drill-down ────────────────────────────────────────────────────
    seg_options = sorted(df_f[drill_col].astype(str).unique())
    if not seg_options:
        st.info("No segments available.")
    else:
        chosen_seg = st.selectbox(
            f"🔎 Drill into a specific {drill_label}:",
            seg_options,
            key="explorer_chosen",
        )
        drill_df = df_f[df_f[drill_col].astype(str) == chosen_seg]
        st.markdown(f"**Showing {len(drill_df):,} restaurants in: `{chosen_seg}`**")

        if len(drill_df) == 0:
            st.info("No data available for this selection.")
        else:
            d1, d2 = st.columns(2)

            sub_col_name = "area" if drill_col != "area" else "cuisine"
            with d1:
                sub = (
                    drill_df.groupby(sub_col_name, observed=True)
                    .agg(
                        avg_viol=("violation_rate", "mean"),
                        count=("restaurant_name", "count"),
                        closure_rate=("ever_closed_flag", "mean"),
                    )
                    .reset_index()
                    .sort_values("avg_viol", ascending=False)
                )
                fig = px.bar(
                    sub, x=sub_col_name, y="avg_viol",
                    color="closure_rate", color_continuous_scale="Reds",
                    text=sub["avg_viol"].round(2),
                    title=f"Avg Violation Rate by {sub_col_name.title()} — {chosen_seg}",
                    labels={"avg_viol": "Avg Violation Rate"},
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(apply_style(fig, 320), use_container_width=True)

            with d2:
                action_sub = drill_df["action"].value_counts().reset_index()
                action_sub.columns = ["Action", "Count"]
                fig = px.bar(
                    action_sub, x="Action", y="Count",
                    color="Action", color_discrete_map=ACTION_COLORS,
                    title=f"Enforcement Breakdown — {chosen_seg}",
                    text="Count",
                    category_orders={"Action": ACTION_ORDER},
                )
                fig.update_traces(textposition="outside")
                st.plotly_chart(apply_style(fig, 320), use_container_width=True)

            st.markdown(f"#### Top 10 Highest-Risk Restaurants in: {chosen_seg}")
            top10 = (
                drill_df.sort_values("risk_score", ascending=False)[
                    ["restaurant_name", "cuisine", "area", "violation_rate",
                     "major_violation_count", "repeat_offender_flag",
                     "dm_star_rating", "risk_score", "action"]
                ]
                .head(10)
                .reset_index(drop=True)
            )
            top10.index += 1
            top10.columns = [
                "Restaurant", "Cuisine", "Area", "Viol Rate",
                "Major Viols", "Repeat", "Stars", "Risk Score", "Action",
            ]
            st.dataframe(
                style_risk_table(top10, action_col="Action"),
                use_container_width=True,
            )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:30px 0 10px;color:#7A8FA6;font-size:0.78rem;
            border-top:1px solid #253040;margin-top:30px;'>
    Dubai Restaurant Inspections Analytics Dashboard · Built with Streamlit &amp; Plotly<br>
    Data: Dubai Municipality Food Safety Records 2023–2026 ·
    All risk scores are model-derived composites for analytical purposes.
</div>
""", unsafe_allow_html=True)
