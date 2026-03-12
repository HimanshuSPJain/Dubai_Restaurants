# 🍽️ Dubai Restaurant Inspections Analytics Dashboard

A four-layer analytics dashboard built with **Streamlit + Plotly + scikit-learn**, exploring Dubai Municipality food safety inspection data (2023–2026).

## 📊 Dashboard Sections

| Tab | Analytics Type | What it answers |
|-----|---------------|-----------------|
| Descriptive | Distribution & overview | What happened? |
| Diagnostic | Correlations & root cause | Why did it happen? |
| Predictive | ML risk model | What will happen? |
| Prescriptive | Recommendations | What should we do? |
| Explorer | Interactive drill-down | Deep-dive any segment |

## 🚀 Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**  
3. Select your repo · Branch: `main` · Main file: `app.py`  
4. Click **Deploy** — done!

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 File Structure

```
├── app.py                    # Main Streamlit dashboard
├── data.csv                  # Dubai restaurant inspection dataset
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme & server config
└── README.md
```

## 🔑 Key Features

- **6 KPI cards** with real-time filter updates  
- **Correlation heatmap** across all numeric features  
- **Random Forest classifier** (AUC > 0.90) with live predictor widget  
- **Sunburst + donut drill-down** — click any cuisine/area/risk tier  
- **Compliance improvement simulator** — model intervention impact  
- **Top 20 priority watchlist** for immediate inspection scheduling  
- Dark luxury theme with gold accents — Syne + DM Sans typography  

## 📦 Dependencies

- `streamlit` — dashboard framework  
- `plotly` — interactive charts  
- `scikit-learn` — Random Forest model  
- `pandas / numpy` — data processing  

## 🎯 Target Variable

`action` — derived from `ever_closed_flag`, `repeat_offender_flag`, and `major_violation_count`:  
- **Closure** — restaurant was closed at least once  
- **Warning Issued** — repeat offender or high major violations  
- **No Action** — compliant restaurant  
