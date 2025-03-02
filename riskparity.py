import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from utils import *
 
def risk_budgeting_portfolio(cov_matrix, b, tol=1e-8, max_iter=1000):
    N = len(b)
    w = np.ones(N) / N  # Initial equal weights

    for _ in range(max_iter):
        w_old = w.copy()

        for i in range(N):
            # Marginal Risk Contribution = dvol / dweight = (Cov * b)_i / (b.T * Cov * b) ** 0.5
            # Risk Contribution = w_i * marginal risk contribution
            # Relative risk contribution = Risk Contribution / Total Risk = w_i * marginal risk contribution / (b.T * Cov * b)    
            total_risk = np.dot(w.T, np.dot(cov_matrix, w))

            # Update weight for asset i
            w_i_new = (b[i] * total_risk) / np.dot(cov_matrix[i, :], w)
            w[i] = max(w_i_new, 0)  # Ensure non-negativity

        # Normalize weights
        w = w / np.sum(w)

        # Convergence check
        if np.linalg.norm(w - w_old) < tol:
            break

    return w

def visualize_rrc(corr_matrix, cov_matrix, w, b, tickers):
    b = b / np.sum(b)

    sigma_w = cov_matrix @ w
    total_risk = w @ sigma_w
    
    # Risk Contributions (RC)
    rc = w * sigma_w
    
    # Relative Risk Contributions (RRC)
    rrc = rc / total_risk

    # Plotting correlation 
    plt.figure(figsize=(8, 6))
    plt.matshow(corr_matrix)  # `fignum=1` ensures it uses the current figure
    plt.colorbar()
    plt.title('Correlation Matrix Heatmap')
    plt.show()


    # Plotting RCC to b
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(w))
    plt.bar(indices - 0.2, rrc, width=0.4, label='Actual RRC')
    plt.bar(indices + 0.2, b, width=0.4, label='Target Risk Budget (b)')
    plt.xlabel('Assets')
    plt.ylabel('Relative Risk Contribution')
    plt.title('Relative Risk Contributions vs Target Risk Budgets')
    plt.xticks(indices, tickers)  # Add ticker labels

    plt.legend()
    plt.grid(axis='y')
    plt.show()


    # Plotting Weights
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(w))
    plt.bar(indices, w, width=0.4, label='Weight of Asset')
    plt.xlabel('Assets')
    plt.ylabel('Weight of Asset')
    plt.title('Relative Risk Contributions vs Target Risk Budgets')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

def produce_rbp_stats(data, b, start_date, end_date):
    # Download historical data
    data = get_trading_start_end_data(start_date, end_date, data)
    # data = data.loc[start_date:end_date, :]
    data = data.pct_change(fill_method=None).dropna()

    # Calculate covariance and correlation matrices
    cov_matrix = data.cov().values

    # Ensure `b` is normalized
    b = b / np.sum(b)

    # Check covariance matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if np.any(eigenvalues < 0):
        print("Covariance matrix is not positive semi-definite. Adjusting...")
        cov_matrix = cov_matrix @ cov_matrix.T  # Make positive semi-definite

    # Calculate weights
    weights = risk_budgeting_portfolio(cov_matrix, b)

    print("Optimized Weights:", weights)
    return weights
    # Visualize results
    # visualize_rrc(corr_matrix, cov_matrix, weights, b, tickers)
   

# produce_rbp_stats(["^GSPC", "^TNX", "GD=F"], np.array([1.4, 1, 1]), start_date="2022-01-01", end_date="2024-12-31")
