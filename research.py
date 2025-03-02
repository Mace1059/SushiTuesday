import riskparity
import mpt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
import pandas as pd
from utils import *
import matplotlib.pyplot as plt


# Tickers: list of asset tickers
# b: RRCs
# year_range_start: date to start from
# years: number of years in long run
def optimal_portfolio_returns(tickers, b, year_range_start, years, short_run_years):

    

    current_date = datetime.strptime(year_range_start, "%Y-%m-%d")
    start_date_for_data = (current_date - relativedelta(months = 25)).strftime("%Y-%m-%d")

    final_date, final_date_str = year(year_range_start, years)
    # print("CURRENT DATE:", year_range_start, "CURRENT DATE TYPE:", type(year_range_start))
    # print("FINAL DATE:", final_date_str, "FINAL DATE TYPE:", type(final_date_str))
    data = yf.download(tickers, start=start_date_for_data, end=final_date_str)["Close"]
    data = data.fillna(method='pad')

    market_data = yf.download("^GSPC", start=start_date_for_data, end=final_date_str)["Close"]

    # find total number of trading days 
    trading_days = get_trading_days(year_range_start, final_date_str)
    daily_rf_rate = 0.045 / trading_days # daily risk free rate


    # Loop through all short term periods, each 5 years on a 6 month sliding window basis
    x_axis_mpt = []
    y_axis_rp = []
    while current_date <= final_date: 
        # print("I LOVE IZZY KAM:", current_date, " - ", final_date)

        end_date, _ = year(current_date, short_run_years)
        print("CURRENT WINDOW:", current_date, " - ", end_date)

        # MPT Portfolio
        sharpe = mpt.find_best_portfolio(data, market_data, daily_rf_rate, current_date, end_date)
        x_axis_mpt.append(sharpe)

        # Ex-Ante Risk Parity Portfolio
        # Loop through every month for 5 years
        current_ex_ante_date = current_date
        total_returns = 0
        total_variance = 0
        while current_ex_ante_date <= end_date and current_ex_ante_date < final_date - relativedelta(days = 32):
          # Use data from two years prior to make predictions
          _, two_years_prior_str = year(current_ex_ante_date, -2)
          end_ex_ante_date = current_ex_ante_date + relativedelta(months = 1)
          print("CURRENT EX-ANTE WINDOW:", current_ex_ante_date, " - ", end_ex_ante_date)
          # Get RRC weights using two years prior until now 
          rp_weights = riskparity.produce_rbp_stats(data, b, two_years_prior_str, current_ex_ante_date.strftime("%Y-%m-%d"))
          # print("WEIGHTS!!!!!", rp_weights)
          # Get returns for the month
          month_returns, month_variance =  mpt.portfolio_returns(current_ex_ante_date.strftime("%Y-%m-%d"), end_ex_ante_date.strftime("%Y-%m-%d"), data, rp_weights)
          total_variance += month_variance 
          total_returns += month_returns
          current_ex_ante_date = current_ex_ante_date + relativedelta(months=1)


        volatility = np.sqrt(total_variance)
        # print("izzy kam returns", total_returns)
        # print("izzy kam vol", volatility)
        total_returns = total_returns
        rp_sharpe = (total_returns - (0.045 / 12)) / volatility
        # print("izzy kam", rp_sharpe)
        print(total_returns, "-", (0.045 / 12), "/", volatility, "=", rp_sharpe)
        y_axis_rp.append(float(rp_sharpe))
        current_date = current_date + relativedelta(months = 6)

    print("Oh my fucking god I'm about to return so hard daddy")
    return x_axis_mpt, y_axis_rp



def rrc_score():
  pass


tickers = ["^GSPC", "^TNX"]

b = np.array([1, 1])

start_date="2000-01-01"

x_axis_mpt, y_axis_rp = optimal_portfolio_returns(tickers, b, start_date, 24, 1)
print(x_axis_mpt, y_axis_rp)
plt.figure()
plt.scatter(x_axis_mpt, y_axis_rp)
# x_line = np.linspace(0, 3, 10)  # Adjust range if needed
# plt.plot(x_line, x_line, color='black', label='y = x')
plt.show()