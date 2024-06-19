# OpenIDX3
Web-based portfolio and stock analysis made for the Indonesian stock market (IDX). 

This program serves as a Stock Portfolio Management System tailored for the Indonesia Stock Exchange (IDX) that enables users to manage their stock portfolio efficiently. The program utilizes yfinance to fetch real-time stock market data and bunch of statistics to do quantitative analysis on their stock portofolio.

## Features

### Portfolio Management

- View an overview of your entire stock portfolio.
- Display key metrics such as portfolio value, market value, and individual stock details including current price, buy price, and buy date.
- Visualize data with line charts and pie charts, including profit/loss information.
- Analyze the returns on your stock investments.
- Manage your stock portfolio with features to add, edit, or delete stocks.
- Validate and process form data to ensure correct input.
- Update portfolio based on user inputs and display appropriate success or error messages.

### Analysis Tools

- Analyze stock ownership distribution.
- Evaluate the performance of selected stocks and display performance metrics and graphs.
- Visualize the distribution of selected stocks.
- Optimize your portfolio for maximum returns or minimum risk, and display results including optimal asset weights for maximum Sharpe ratio and minimum volatility.
- Perform fundamental analysis on selected stocks and display fundamental data such as P/E ratios, earnings reports, and more.
- Decompose stock prices into trend and seasonal components.
- Apply GARCH (Generalized Auto Regressive Conditional Heteroskedasticity) models to analyze stock price volatility and display GARCH model outputs and graphs.

### Prediction Tools
- News sentiment analysis using DistilRoBERTa
- Price prediction using LSTM (Long Short-Term Memory) or ARIMA (Auto Regressive Integrated Moving Average) model

## Installation

1. Clone the repository: `git clone https://github.com/openidx-team/OpenIDX3.git`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

1. Navigate to the project directory: `cd OpenIDX3`
2. Run the program: `python main.py`

## License

This project is licensed under the [GPL-3.0 License](LICENSE).

## Disclaimer

The information provided in this program is for general informational purposes only. It should not be considered as financial or investment advice. The user of this program is solely responsible for any investment decisions made based on the information provided. The developers of this program do not guarantee the accuracy, completeness, or reliability of the information. Use this program at your own risk.
