# 🍽️ Dubai Restaurant Health Violations Dashboard

A production-ready Streamlit analytics dashboard for predicting and analyzing restaurant health violations across Dubai (2023–2026).

## 🚀 Live Demo
Deploy instantly to [Streamlit Cloud](https://streamlit.io/cloud) — free tier compatible.

---

## 📦 Project Structure

```
dubai_dashboard/
├── app.py                    # Main Streamlit app (all 5 tabs)
├── utils.py                  # Data loading, preprocessing, KPI helpers
├── models.py                 # ML training, evaluation, SHAP
├── data.csv                  # Dubai restaurant violations dataset
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dubai-themed color scheme
└── README.md
```

---

## 🎛️ Dashboard Sections

| Tab | Content |
|-----|---------|
| 🏠 Home | Executive summary, KPI metrics, trend overview |
| 📊 Descriptive | Dataset overview, distributions, heatmaps |
| 🔬 Diagnostic | Correlation analysis, violation drivers, area profiles |
| 🤖 Predictive | 5 ML models, ROC curves, SHAP explanations |
| 🎯 Prescriptive | Risk scoring, prioritization matrix, action plans |
| 🏆 Leaderboard | Top violators table with sorting & download |

---

## 🤖 ML Models Included

- **Logistic Regression** (baseline)
- **Random Forest** (200 trees)
- **XGBoost**
- **LightGBM**
- **Gradient Boosting**

Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV

---

## 🚀 Deployment Instructions

### Option 1: Streamlit Cloud (Recommended)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repo → set **Main file path** to `app.py`
5. Click **Deploy** — done in ~2 minutes!

> ⚠️ Make sure `data.csv` is in the **root of the repo** alongside `app.py`

### Option 2: Run Locally

```bash
# Clone / navigate to the project folder
cd dubai_dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## 📊 Target Variable

**`ever_closed_flag`** — binary (0 = never closed, 1 = closed at least once)

Used as the ML prediction target. A restaurant being closed indicates a serious health violation that triggered enforcement action.

---

## 🎨 Design

- **Color palette**: Dubai gold (#C9A84C), Navy (#0A1628), Teal (#00B4D8)
- **Charting**: Plotly (interactive, dark theme)
- **Responsive** sidebar filters apply across all tabs

---

## ⚙️ Configuration

Edit `.streamlit/config.toml` to customize the theme:

```toml
[theme]
primaryColor = "#C9A84C"      # Gold
backgroundColor = "#0A1628"   # Dark navy
secondaryBackgroundColor = "#112240"
textColor = "#E8EAF0"
```

---

## 📋 Data Dictionary

| Column | Description |
|--------|-------------|
| `license_no` | Unique restaurant identifier |
| `restaurant_name` | Restaurant name |
| `cuisine` | Cuisine type |
| `area` | Dubai district |
| `emirate` | Emirate (all Dubai) |
| `inspection_count_2023_2026` | Number of inspections |
| `violation_count_2023_2026` | Total violations found |
| `major_violation_count` | Count of major/critical violations |
| `repeat_offender_flag` | 1 if repeat offender |
| `closure_events` | Number of closure incidents |
| `ever_closed_flag` | **Target**: 1 if restaurant was ever closed |
| `dm_star_rating` | Dubai Municipality star rating (1–5) |
| `last_inspection_date` | Date of most recent inspection |

---

*Built for Dubai Municipality Health & Safety Analytics · 2024*
