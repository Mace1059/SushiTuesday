import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Step 1: Fetch Stock Data
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
start_date = "2020-01-01"
end_date = "2024-06-01"

# Download stock data
data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
returns = data.pct_change().dropna()  # Daily returns

# Step 2: Portfolio Metrics
def portfolio_performance(weights, mean_returns, cov_matrix):
    p_return = np.sum(mean_returns * weights) * 252  # Annualized Return
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized Volatility
    return p_return, p_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility  # Negative Sharpe to minimize

def min_variance(mean_returns, cov_matrix):
    def portfolio_volatility(weights):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    initial_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

    result = minimize(portfolio_volatility, initial_weights, bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, target_returns):
    frontier = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[0] - target}
        ]
        bounds = tuple((0, 1) for _ in range(len(mean_returns)))
        initial_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

        result = minimize(lambda w: portfolio_performance(w, mean_returns, cov_matrix)[1],
                          initial_weights, bounds=bounds, constraints=constraints)
        if result.success:
            frontier.append(result.fun)
        else:
            frontier.append(None)
    return frontier

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mean_returns)))
    initial_weights = np.array([1 / len(mean_returns)] * len(mean_returns))

    result = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns, cov_matrix, risk_free_rate),
                      bounds=bounds, constraints=constraints)
    return result

# Step 3: Perform Calculations
mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.02  # Risk-free rate

# Minimum Variance Portfolio
min_var_port = min_variance(mean_returns, cov_matrix)
min_var_return, min_var_volatility = portfolio_performance(min_var_port.x, mean_returns, cov_matrix)

# Max Sharpe Ratio Portfolio
max_sharpe_port = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
max_sharpe_return, max_sharpe_volatility = portfolio_performance(max_sharpe_port.x, mean_returns, cov_matrix)

# Efficient Frontier
target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, 50)
frontier_volatilities = efficient_frontier(mean_returns, cov_matrix, target_returns)

# Step 4: Plot Results
plt.figure(figsize=(10, 6))

# Plot individual stocks
for stock in tickers:
    plt.scatter(np.sqrt(returns[stock].var() * 252), returns[stock].mean() * 252, label=stock)

# Plot efficient frontier
plt.plot(frontier_volatilities, target_returns, linestyle='--', color='black', label='Efficient Frontier')

# Plot minimum variance portfolio
plt.scatter(min_var_volatility, min_var_return, color='red', marker='*', s=200, label='Minimum Variance Portfolio')

# Plot maximum Sharpe ratio portfolio
plt.scatter(max_sharpe_volatility, max_sharpe_return, color='blue', marker='*', s=200, label='Max Sharpe Ratio Portfolio')

plt.title('Efficient Frontier with Optimal Portfolios')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Print Results
print("Minimum Variance Portfolio Weights:", min_var_port.x)
print(f"Minimum Variance Portfolio Return: {min_var_return:.2f}, Volatility: {min_var_volatility:.2f}")

print("\nMaximum Sharpe Ratio Portfolio Weights:", max_sharpe_port.x)
print(f"Max Sharpe Portfolio Return: {max_sharpe_return:.2f}, Volatility: {max_sharpe_volatility:.2f}")
print(f"Sharpe Ratio: {((max_sharpe_return - risk_free_rate) / max_sharpe_volatility):.2f}")
