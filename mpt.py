import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from utils import *
pd.options.display.max_rows = 1000


def portfolio_returns(start_date, end_date, data, weights):
    data = get_trading_start_end_data(start_date, end_date, data)
    # data = data.loc[start_date:end_date, :]
    weights = weights / np.sum(weights)
    daily_returns = data.pct_change(fill_method=None).dropna()
    # print("shape returns", np.shape(daily_returns))
    # print("shape weights", np.shape(weights))
    cov_matrix = daily_returns.cov()
    # Calculate portfolio variance using the quadratic form
    variance = weights.T @ cov_matrix @ weights
    
    returns = daily_returns @ weights

    total_return = (returns + 1).dropna().cumprod()

    # print("fuck you", total_return)

    return total_return.iloc[-1] - 1, variance * 21


def expected_portfolio_performance(weights, mean_returns, cov_matrix, trading_days):
    p_return = np.dot(mean_returns, weights) * trading_days  # Annualized Return
    p_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(trading_days)  # Annualized Volatility
    return p_return, p_volatility

def negative_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate, trading_days):
    p_return, p_volatility = expected_portfolio_performance(weights, expected_returns, cov_matrix, trading_days)
    return - (p_return - risk_free_rate)/(p_volatility)

def efficient_frontier(expected_returns, cov_matrix, target_returns, trading_days):
    frontier = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: expected_portfolio_performance(x, expected_returns, cov_matrix, trading_days)[0] - target}
        ]
        bounds = tuple((0, 1) for _ in range(len(expected_returns)))
        initial_weights = np.array([1 / len(expected_returns)] * len(expected_returns))

        result = minimize(lambda w: expected_portfolio_performance(w, expected_returns, cov_matrix, trading_days)[1],
                          initial_weights, bounds=bounds, constraints=constraints)
        if result.success:
            frontier.append(result.fun)
        else:
            frontier.append(None)
    return [volatility for volatility in frontier if volatility is not None]  # Filter out None values


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, trading_days):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    initial_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

    result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate, trading_days),
                      bounds=bounds, constraints=constraints)
    return result

def calculate_beta(asset_returns, market_returns):
    covariance = np.cov(asset_returns, market_returns)[0][1]
    # print(covariance)
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    return beta

def find_best_portfolio(data, market_data, risk_free_rate, start_date, end_date):
    data = data.loc[start_date:end_date, :]

    # Download stock data
    new_tickers = data.columns.tolist()

    # Download market data????
    market_data = market_data.loc[start_date:end_date, :]

    # Get Daily returns
    daily_stock_returns = data.pct_change().dropna()
    cov_daily = daily_stock_returns.cov()

    market_returns = market_data.pct_change().dropna().iloc[:, 0]

    trading_days = get_trading_days(start_date, end_date)

    # Calculate betas
    betas = {}
    for ticker in new_tickers:
        betas[ticker] = calculate_beta(daily_stock_returns[ticker], market_returns)

    # Apply CAPM model
    # risk_free_rate = 0.04366 / 252 # Risk-free rate
    daily_expected_market_return = ((market_returns + 1).dropna().cumprod().iloc[-1] - 1) / trading_days
    daily_rfr = risk_free_rate / trading_days
    # annualized_expected_market_return = market_returns.mean()
    market_risk_premium = (daily_expected_market_return - daily_rfr)
    expected_returns = np.array([daily_rfr + beta * (market_risk_premium) for beta in betas.values()])

    # Efficient Frontier
    target_returns = np.linspace(expected_returns.min() * trading_days, expected_returns.max() * trading_days, 50)
    frontier_volatilities = efficient_frontier(expected_returns, cov_daily, target_returns, trading_days)

    # Max sharpe
    max_sharpe_portfolio = max_sharpe_ratio(expected_returns, cov_daily, daily_rfr, trading_days)
    max_sharpe_return, max_sharpe_volatility = expected_portfolio_performance(max_sharpe_portfolio.x, expected_returns, cov_daily, trading_days)
    df = pd.DataFrame(max_sharpe_portfolio.x, index=new_tickers, columns=["Value"])
    pd.set_option('display.float_format', '{:.6f}'.format)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(max_sharpe_volatility, max_sharpe_return, color='blue', s=200, label='Max Sharpe Ratio Portfolio')
    sharpe = (max_sharpe_return - daily_rfr) / max_sharpe_volatility
    # x = np.linspace(0, frontier_volatilities[-1], 100)
    # x = np.linspace(0, frontier_volatilities[-1], 100)
    # y = x * sharpe + risk_free_rate
    # plt.plot(x, y)
    # plt.plot(frontier_volatilities, target_returns, color='black', label='Efficient Frontier')
    return sharpe