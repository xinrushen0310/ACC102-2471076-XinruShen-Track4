# Manchester United Stock & Match Performance Analysis

## 1. Problem & User
This project investigates whether Manchester United's (NYSE: MANU) on-pitch performance in the Premier League has a measurable impact on its short-term stock price movements. The intended audience includes sports economists, financial analysts, and retail investors interested in the intersection of sports performance and financial markets.

## 2. Data
The analysis integrates two primary datasets covering six Premier League seasons (August 2019 to May 2025):
- **Stock Data:** Monthly stock prices, trading volumes, and returns for MANU, sourced from the Center for Research in Security Prices (CRSP) via the Wharton Research Data Services (WRDS) platform (accessed April 2025).
- **Match Data:** Match-level results (goals, outcomes, and points) sourced from [football-data.org](https://www.football-data.org/) (accessed April 2025).

## 3. Methods
The analytical workflow was implemented entirely in Python:
- **Data Acquisition:** Queried WRDS PostgreSQL database using the `wrds` library and fetched CSV files via `requests` and `pandas`.
- **Data Cleaning & Transformation:** Parsed match dates, aggregated results to a monthly frequency, and determined the dominant match result (Win, Draw, Loss) for each month.
- **Analysis & Visualization:** Utilized `matplotlib` and `seaborn` for time-series plots, box plots, and scatter plots. Conducted statistical testing (independent t-tests and Pearson correlation) using `scipy.stats`.
- **Productization:** Built an interactive web application using `streamlit` to allow users to explore the data dynamically.

## 4. Key Findings
- **Win vs. Loss Returns:** Months dominated by wins show a slightly higher average stock return compared to loss-dominated months, but the difference is not statistically significant at the 5% level.
- **Goal Difference Correlation:** There is a very weak positive relationship (Pearson r ≈ 0.15) between monthly goal difference and stock returns, indicating that large goal differences do not reliably predict positive stock performance.
- **Corporate Events Dominate:** Macro-economic and corporate events overwhelmingly dictate price movements. The sharpest decline occurred in March 2020 (COVID-19), while the most significant surge (+40%) happened in early 2023 amid rumors of a club sale.

## 5. How to Run
To run the Streamlit application locally:

1. Clone this repository or download the project files.
2. Ensure you have Python 3.7+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
5. Open the provided local URL (usually `http://localhost:8501`) in your web browser.

## 6. Product Link / Demo
- **Streamlit App:** [(https://acc102-2471076-xinrushen-track4-v93c5ubfz5cydcbuljra2k.streamlit.app/)]
- **Demo Video:** [Insert your 1-3 minute demo video link here]

## 7. Limitations & Next Steps
- **Limitations:** The primary limitation is the mismatch in data frequency. Aggregating weekly match results to a monthly level obscures the immediate market reaction to specific victories or defeats. The analysis also does not control for broader market trends or financial fundamentals.
- **Next Steps:** Future improvements could involve utilizing daily stock data and conducting an event study methodology to measure abnormal returns in the days immediately following a match. Incorporating sentiment analysis from social media or news articles could also provide a more nuanced view of investor reactions.
