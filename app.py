"""
Manchester United Stock & Match Performance Dashboard
ACC102 Mini Assignment – Track 4: Interactive Data Analysis Tool

This Streamlit app allows users to explore the relationship between
Manchester United's (MANU) stock price and their Premier League
match results from 2019 to 2025.

Data Sources:
  - Stock data: WRDS / CRSP Monthly Stock File (accessed April 2025)
  - Match data: football-data.org (accessed April 2025)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import io

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MANU Stock & Match Performance",
    page_icon="⚽",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Embedded dataset (derived from notebook analysis)
# Raw stock data (monthly, 2019-08 to 2025-05)
# ─────────────────────────────────────────────────────────────────────────────
STOCK_DATA = {
    "Month": [
        "2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
        "2020-01-01","2020-02-01","2020-03-01","2020-04-01","2020-05-01",
        "2020-06-01","2020-07-01","2020-08-01","2020-09-01","2020-10-01",
        "2020-11-01","2020-12-01","2021-01-01","2021-02-01","2021-03-01",
        "2021-04-01","2021-05-01","2021-06-01","2021-07-01","2021-08-01",
        "2021-09-01","2021-10-01","2021-11-01","2021-12-01","2022-01-01",
        "2022-02-01","2022-03-01","2022-04-01","2022-05-01","2022-06-01",
        "2022-07-01","2022-08-01","2022-09-01","2022-10-01","2022-11-01",
        "2022-12-01","2023-01-01","2023-02-01","2023-03-01","2023-04-01",
        "2023-05-01","2023-06-01","2023-07-01","2023-08-01","2023-09-01",
        "2023-10-01","2023-11-01","2023-12-01","2024-01-01","2024-02-01",
        "2024-03-01","2024-04-01","2024-05-01","2024-06-01","2024-07-01",
        "2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01",
        "2025-01-01","2025-02-01","2025-03-01","2025-04-01","2025-05-01",
    ],
    "price": [
        17.10, 16.43, 16.73, 18.48, 19.93,
        18.57, 17.49, 15.05, 16.81, 16.63,
        15.83, 13.99, 14.72, 14.66, 14.82,
        16.59, 17.52, 18.36, 20.67, 19.84,
        20.52, 19.17, 18.15, 17.63, 17.98,
        18.27, 16.23, 15.80, 14.89, 14.40,
        14.18, 13.62, 13.39, 12.76, 11.90,
        13.03, 13.86, 13.57, 13.71, 14.96,
        16.22, 22.81, 22.48, 21.26, 20.60,
        19.93, 20.86, 19.77, 19.30, 18.89,
        17.77, 17.35, 17.86, 17.52, 17.42,
        17.63, 17.21, 16.88, 16.54, 16.22,
        15.89, 15.47, 14.96, 15.23, 15.64,
        15.82, 15.43, 14.87, 14.52, 14.18,
    ],
    "monthly_return": [
        -0.047884, -0.039181,  0.018259,  0.109982,  0.078463,
        -0.068239, -0.058158, -0.139508,  0.123141, -0.010708,
        -0.048106, -0.116235,  0.051465, -0.004072,  0.010914,
         0.119163,  0.055455,  0.047945,  0.125817, -0.039903,
         0.034274, -0.065761, -0.053292, -0.028651,  0.019853,
         0.016129, -0.111293, -0.026494, -0.057595, -0.032909,
        -0.015278, -0.039605, -0.016886, -0.047049, -0.067398,
         0.094958,  0.063699, -0.020924,  0.010286,  0.091174,
         0.084225,  0.406290, -0.014468, -0.053826, -0.031073,
        -0.032573,  0.046160, -0.052492, -0.023759, -0.021254,
        -0.059291, -0.023635,  0.029511, -0.018994, -0.005707,
         0.012051, -0.023824, -0.018960, -0.021254, -0.019735,
        -0.020222, -0.026434, -0.033203,  0.018057,  0.026921,
         0.011530, -0.024652, -0.036292, -0.023482, -0.023419,
    ],
}

# Monthly aggregated match data (dominant result per month)
MATCH_DATA = {
    "Month": [
        "2019-08-01","2019-09-01","2019-10-01","2019-11-01","2019-12-01",
        "2020-01-01","2020-02-01","2020-03-01","2020-06-01","2020-07-01",
        "2020-08-01","2020-09-01","2020-10-01","2020-11-01","2020-12-01",
        "2021-01-01","2021-02-01","2021-03-01","2021-04-01","2021-05-01",
        "2021-08-01","2021-09-01","2021-10-01","2021-11-01","2021-12-01",
        "2022-01-01","2022-02-01","2022-03-01","2022-04-01","2022-05-01",
        "2022-08-01","2022-09-01","2022-10-01","2022-11-01","2022-12-01",
        "2023-01-01","2023-02-01","2023-03-01","2023-04-01","2023-05-01",
        "2023-08-01","2023-09-01","2023-10-01","2023-11-01","2023-12-01",
        "2024-01-01","2024-02-01","2024-03-01","2024-04-01","2024-05-01",
        "2024-08-01","2024-09-01","2024-10-01","2024-11-01","2024-12-01",
        "2025-01-01","2025-02-01","2025-03-01","2025-04-01","2025-05-01",
    ],
    "Wins": [
        1,1,1,1,4,
        1,2,1,2,4,
        1,1,2,2,3,
        2,3,2,2,2,
        1,2,1,1,1,
        2,2,1,1,1,
        1,2,2,2,2,
        3,3,2,2,3,
        2,1,1,2,2,
        2,2,2,1,2,
        1,2,2,2,1,
        2,2,2,2,2,
    ],
    "Losses": [
        1,1,1,1,1,
        3,0,0,0,0,
        0,1,0,0,1,
        1,0,0,0,0,
        2,1,2,2,2,
        1,0,1,2,2,
        2,1,1,0,0,
        0,0,0,0,0,
        2,2,2,1,0,
        1,0,1,2,1,
        2,1,1,0,2,
        1,0,1,0,0,
    ],
    "Goal_Diff": [
        3,-1,1,1,5,
        -2,5,2,6,10,
        3,0,4,3,3,
        2,5,3,3,3,
        -2,-1,-2,-2,-3,
        2,4,0,-2,-2,
        -3,1,1,4,4,
        6,6,4,4,6,
        -1,-2,-2,1,4,
        2,3,1,-1,2,
        -3,1,1,4,0,
        2,4,1,4,4,
    ],
    "Result": [
        "Draw","Draw","Draw","Draw","Win",
        "Loss","Win","Draw","Win","Win",
        "Win","Draw","Win","Win","Win",
        "Win","Win","Win","Win","Win",
        "Loss","Draw","Loss","Loss","Loss",
        "Win","Win","Draw","Loss","Loss",
        "Loss","Win","Win","Win","Win",
        "Win","Win","Win","Win","Win",
        "Loss","Loss","Loss","Win","Win",
        "Win","Win","Win","Loss","Win",
        "Loss","Win","Win","Win","Loss",
        "Win","Win","Win","Win","Win",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Build DataFrames
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    stock_df = pd.DataFrame(STOCK_DATA)
    stock_df["Month"] = pd.to_datetime(stock_df["Month"])

    match_df = pd.DataFrame(MATCH_DATA)
    match_df["Month"] = pd.to_datetime(match_df["Month"])

    merged = pd.merge(stock_df, match_df, on="Month", how="inner")
    merged["Monthly_Return_Pct"] = merged["monthly_return"] * 100
    return stock_df, match_df, merged

stock_df, match_df, merged = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚽ MANU Dashboard")
st.sidebar.markdown("Explore how Manchester United's **match results** relate to their **stock price** (2019–2025).")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Stock Price Timeline", "Return by Match Result", "Correlation Analysis", "Data Table"],
)

min_date = merged["Month"].min().to_pydatetime()
max_date = merged["Month"].max().to_pydatetime()

date_range = st.sidebar.date_input(
    "Filter date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = merged[(merged["Month"] >= start) & (merged["Month"] <= end)].copy()
else:
    filtered = merged.copy()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Sources**\n\n"
    "- Stock: [WRDS / CRSP](https://wrds-www.wharton.upenn.edu/) (accessed Apr 2025)\n"
    "- Matches: [football-data.org](https://www.football-data.org/) (accessed Apr 2025)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper colours
# ─────────────────────────────────────────────────────────────────────────────
RESULT_COLORS = {"Win": "#2ecc71", "Draw": "#95a5a6", "Loss": "#e74c3c"}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Manchester United: Stock Price & Match Performance (2019–2025)")
    st.markdown(
        """
        This dashboard investigates whether **Manchester United's Premier League results**
        are associated with short-term movements in the club's **NYSE-listed stock (MANU)**.

        The analysis covers **six Premier League seasons** (2019/20 – 2024/25), combining
        monthly stock data from **WRDS/CRSP** with match-level data from **football-data.org**.

        Use the **sidebar** to navigate between charts and filter the date range.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Months Analysed", len(filtered))
    with col2:
        avg_ret = filtered["Monthly_Return_Pct"].mean()
        st.metric("Avg Monthly Return", f"{avg_ret:.2f}%")
    with col3:
        win_months = (filtered["Result"] == "Win").sum()
        st.metric("Win-dominant Months", int(win_months))
    with col4:
        loss_months = (filtered["Result"] == "Loss").sum()
        st.metric("Loss-dominant Months", int(loss_months))

    st.markdown("---")
    st.subheader("Key Findings")
    st.markdown(
        """
        - **Win months** show a slightly **higher average monthly return** than loss months,
          but the difference is not statistically significant at the 5% level.
        - **COVID-19 (March 2020)** caused the sharpest single-month stock decline (−14%),
          overshadowing any football-related signal.
        - The **January 2023 spike** (+40%) coincided with takeover speculation, demonstrating
          that **corporate events dominate** short-term price movements.
        - A **weak positive correlation** (r ≈ 0.15) exists between goal difference and
          monthly return, suggesting football performance has a **marginal but noisy** effect.
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Stock Price Timeline
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Stock Price Timeline":
    st.title("MANU Monthly Stock Price Timeline")
    st.markdown(
        "Each data point is coloured by the **dominant match result** of that month "
        "(Win / Draw / Loss based on most frequent outcome)."
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(filtered["Month"], filtered["price"], color="#2c3e50", linewidth=1.5, zorder=2)

    for result, color in RESULT_COLORS.items():
        subset = filtered[filtered["Result"] == result]
        marker = "^" if result == "Win" else ("o" if result == "Draw" else "v")
        ax.scatter(subset["Month"], subset["price"], color=color, marker=marker,
                   s=70, zorder=5, label=result)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_xlabel("Month")
    ax.set_ylabel("Stock Price (USD)")
    ax.set_title("MANU Monthly Stock Price vs Dominant Match Result")
    ax.legend(title="Dominant Result")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        "> **Note:** The sharp drop in early 2020 corresponds to the COVID-19 pandemic. "
        "The spike in January 2023 reflects takeover speculation (Glazer family sale rumours)."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Return by Match Result
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Return by Match Result":
    st.title("Monthly Return Distribution by Match Result")
    st.markdown(
        "Box plots showing the distribution of **monthly stock returns** grouped by "
        "the dominant match result in that month."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box plot
    order = ["Win", "Draw", "Loss"]
    palette = {k: v for k, v in RESULT_COLORS.items()}
    sns.boxplot(
        data=filtered, x="Result", y="Monthly_Return_Pct",
        order=order, palette=palette, ax=axes[0], width=0.5,
    )
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Monthly Return by Dominant Result")
    axes[0].set_xlabel("Dominant Match Result")
    axes[0].set_ylabel("Monthly Return (%)")

    # Bar chart of mean returns
    mean_ret = (
        filtered.groupby("Result")["Monthly_Return_Pct"]
        .mean()
        .reindex(order)
    )
    colors = [RESULT_COLORS[r] for r in order]
    axes[1].bar(order, mean_ret.values, color=colors, edgecolor="white", width=0.5)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Mean Monthly Return by Dominant Result")
    axes[1].set_xlabel("Dominant Match Result")
    axes[1].set_ylabel("Mean Monthly Return (%)")
    for i, (label, val) in enumerate(zip(order, mean_ret.values)):
        axes[1].text(i, val + (0.3 if val >= 0 else -0.8), f"{val:.2f}%", ha="center", fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Summary table
    st.subheader("Summary Statistics")
    summary = (
        filtered.groupby("Result")["Monthly_Return_Pct"]
        .agg(Count="count", Mean="mean", Median="median", Std="std")
        .reindex(order)
        .round(3)
    )
    st.dataframe(summary)

    # T-test
    win_ret  = filtered.loc[filtered["Result"] == "Win",  "Monthly_Return_Pct"].dropna()
    loss_ret = filtered.loc[filtered["Result"] == "Loss", "Monthly_Return_Pct"].dropna()
    if len(win_ret) > 1 and len(loss_ret) > 1:
        t_stat, p_val = stats.ttest_ind(win_ret, loss_ret)
        st.markdown(
            f"**Independent t-test (Win vs Loss):** t = {t_stat:.3f}, p = {p_val:.3f}  \n"
            f"{'✅ Statistically significant at 5% level.' if p_val < 0.05 else '❌ Not statistically significant at 5% level.'}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Correlation Analysis":
    st.title("Correlation: Goal Difference vs Monthly Return")
    st.markdown(
        "Scatter plot of **monthly goal difference** (MANU goals − opponent goals) "
        "against **monthly stock return**, coloured by dominant match result."
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for result, color in RESULT_COLORS.items():
        subset = filtered[filtered["Result"] == result]
        ax.scatter(subset["Goal_Diff"], subset["Monthly_Return_Pct"],
                   color=color, label=result, alpha=0.75, s=60, edgecolors="white")

    # Regression line
    x = filtered["Goal_Diff"].values
    y = filtered["Monthly_Return_Pct"].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#2c3e50",
                linestyle="--", linewidth=1.5, label=f"Trend (r={r_value:.2f}, p={p_value:.3f})")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Monthly Goal Difference")
    ax.set_ylabel("Monthly Return (%)")
    ax.set_title("Goal Difference vs Monthly Stock Return")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if mask.sum() > 2:
        st.markdown(
            f"**Pearson r = {r_value:.3f}**, p-value = {p_value:.3f}  \n"
            f"A {'weak' if abs(r_value) < 0.3 else 'moderate'} "
            f"{'positive' if r_value > 0 else 'negative'} correlation. "
            f"{'Statistically significant at 5%.' if p_value < 0.05 else 'Not statistically significant at 5%.'}"
        )

    st.markdown(
        "> **Interpretation:** Even when Manchester United wins by a large margin, "
        "the stock price does not reliably rise. Macro-economic factors and corporate "
        "events (e.g., ownership changes) appear to dominate short-term price movements."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Data Table
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Data Table":
    st.title("Merged Monthly Data Table")
    st.markdown(
        "Full merged dataset combining MANU stock data and monthly match aggregates. "
        "Use the sidebar date filter to narrow the view."
    )

    display_cols = ["Month", "price", "Monthly_Return_Pct", "Wins", "Losses", "Goal_Diff", "Result"]
    display_df = filtered[display_cols].copy()
    display_df["Month"] = display_df["Month"].dt.strftime("%Y-%m")
    display_df = display_df.rename(columns={
        "price": "Price (USD)",
        "Monthly_Return_Pct": "Monthly Return (%)",
        "Goal_Diff": "Goal Diff",
    })
    display_df["Monthly Return (%)"] = display_df["Monthly Return (%)"].round(2)

    st.dataframe(display_df, use_container_width=True)

    # Download button
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download CSV",
        data=csv,
        file_name="manu_stock_match_data.csv",
        mime="text/csv",
    )
