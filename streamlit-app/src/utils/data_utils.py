import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm

def fetch_fmp_data(ticker, start, end):
    """
    Fetch historical adjusted close prices from Financial Modeling Prep API.
    
    Parameters:
    - ticker: str, the ticker symbol (e.g., 'AAPL')
    - start: str, start date in format 'YYYY-MM-DD'
    - end: str, end date in format 'YYYY-MM-DD'
    - api_key: str, your FMP API key
    
    Returns:
    - pd.Series: A pandas Series of adjusted close prices indexed by date.
    """
    api_key = "koXQKn9g5d174jJu9WH3MgX8sVGQdBOi"
    
    # This endpoint fetches daily historical price data for the specified date range
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}&from={start}&to={end}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}, {response.text}")
    
    data = response.json()
    if "historical" not in data:
        raise ValueError(f"No historical data found for {ticker}")
    
    # Extract adjusted close prices
    historical = data["historical"]
    prices = {item["date"]: item["adjClose"] for item in historical}
    
    # Convert to pandas Series
    return pd.Series(prices).sort_index()

import requests

def fetch_expense_ratio(ticker):
    """
    Fetch the expense ratio for a mutual fund or ETF using the FMP API.

    Parameters:
    - ticker: str, the ticker symbol of the mutual fund (e.g., 'VFIAX')
    - api_key: str, your FMP API key

    Returns:
    - float: The expense ratio of the mutual fund, or None if not available.
    """
    api_key = "koXQKn9g5d174jJu9WH3MgX8sVGQdBOi"
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}, {response.text}")

    data = response.json()
    if not data or "expenseRatio" not in data[0]:
        print(f"Expense ratio not available for {ticker}")
        return None

    # Extract the expense ratio
    expense_ratio = data[0].get("expenseRatio")
    return expense_ratio



def get_all_returns(tickers, start='2024-01-01', end='2025-01-01'):
    """
    Fetch historical data and calculate individual returns for each ticker.

    Parameters:
    - tickers: list of str, the ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    - start: str, start date in format 'YYYY-MM-DD'
    - end: str, end date in format 'YYYY-MM-DD'
    - api_key: str, your FMP API key

    Returns:
    - list of pd.Series: A list of individual returns for each ticker.
    """
    all_returns = []  # List to store individual returns for each ticker

    for ticker in tickers:
        try:
            # Fetch historical adjusted close prices
            data = fetch_fmp_data(ticker, start, end)

            # Calculate daily returns
            returns = data.pct_change().dropna()
            all_returns.append(returns)  # Store returns for this ticker

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            all_returns.append(None)  # Append None for tickers with errors

    return all_returns

def calculate_correlation_matrix(data):
    """
    Calculate the correlation matrix for the given data.

    Parameters:
    data (pd.DataFrame): DataFrame containing historical prices.

    Returns:
    np.array: Correlation matrix.
    """
    returns = data.pct_change().dropna()
    correlation_matrix = returns.corr().values
    return correlation_matrix

def calculate_diversification_factor(weights, correlation_matrix):
    """
    Calculate the portfolio diversification factor (PDF).

    Parameters:
    weights (list or np.array): The weights of the assets in the portfolio.
    correlation_matrix (np.array): The correlation matrix of the assets.

    Returns:
    float: The diversification factor of the portfolio.
    """
    # Ensure weights and correlation matrix are numpy arrays
    weights = np.array(weights)
    correlation_matrix = np.array(correlation_matrix)

    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(correlation_matrix, weights))

    # Calculate the weighted variance (assuming all variances are 1 for simplicity)
    weighted_variance = np.sum(weights ** 2)

    # Diversification factor
    diversification_factor = portfolio_variance / weighted_variance

    return diversification_factor


def calculate_metrics(all_returns, tickers, rf_rate=0.01):
    """
    Calculate metrics for individual tickers using their returns, including alpha and beta.

    Parameters:
    - all_returns: list of pd.Series, individual returns for each ticker.
    - tickers: list of str, the ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
    - rf_rate: float, annual risk-free rate (default = 0.01 for 1%).

    Returns:
    - pd.DataFrame: A DataFrame containing metrics for each ticker.
    """
    results = []  # List to store results for each ticker

    # Hardcoded benchmark
    benchmark = 'SPY'

    # Fetch benchmark data
    benchmark_data = fetch_fmp_data(benchmark, start='2024-01-01', end='2025-01-01')
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Convert annual risk-free rate to daily
    rf_daily = rf_rate / 252

    # Calculate benchmark excess returns
    benchmark_excess_returns = benchmark_returns - rf_daily

    for ticker, returns in zip(tickers, all_returns):
        if returns is None:
            # Handle cases where returns could not be calculated
            results.append({
                "Ticker": ticker,
                "Alpha": None,
                "Beta": None,
                "Volatility": None,
                "Sharpe Ratio": None,
                "Calmar Ratio": None,
                "Annualized Return": None,
                "Sortino Ratio": None,
                "Expense Ratio": None,
                "Error": "Data not available"
            })
            continue

        # Calculate excess returns
        excess_returns = returns - rf_daily

        # Calculate beta using covariance and variance
        covariance = np.cov(excess_returns, benchmark_excess_returns)[0, 1]
        variance = np.var(benchmark_excess_returns)
        beta = covariance / variance

        # Calculate alpha
        alpha = np.mean(excess_returns) - beta * np.mean(benchmark_excess_returns)

        # Calculate other metrics
        vol = np.std(returns.values)
        sharpe_ratio = np.mean(returns.values - rf_daily) / vol
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        annual_return = (1 + returns.mean()) ** 12 - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        downside_returns = returns[returns < rf_daily]
        downside_deviation = np.std(downside_returns.values)
        sortino_ratio = np.mean(returns.values - rf_daily) / downside_deviation if downside_deviation != 0 else np.nan

        # Fetch expense ratio
        expense_ratio = fetch_expense_ratio(ticker)

        # Append results for this ticker
        results.append({
            "Ticker": ticker,
            "Alpha": f"{alpha * 100:.2f}%",  # Format as percentage
            "Beta": beta,  # Keep as a ratio
            "Volatility": f"{vol * 100:.2f}%",  # Format as percentage
            "Sharpe Ratio": sharpe_ratio,  # Keep as a ratio
            "Calmar Ratio": calmar_ratio,  # Keep as a ratio
            "Annualized Return": f"{annual_return * 100:.2f}%",  # Format as percentage
            "Sortino Ratio": sortino_ratio,  # Keep as a ratio
            "Expense Ratio": f"{expense_ratio * 100:.2f}%" if expense_ratio is not None else "N/A"  # Format as percentage
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
  returns = np.sum(mean_returns*weights) * 252 # Number of trading days in the year
  std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
  return std, returns



def calculate_portfolio_metrics(tickers, all_returns, rf_rate=0.01):
    """
    Calculate metrics for an equally weighted portfolio.

    Parameters:
    - all_returns: list of pd.Series, individual returns for each ticker.
    - rf_rate: float, annual risk-free rate (default = 0.01 for 1%).

    Returns:
    - dict: A dictionary containing portfolio metrics.
    """
    
    num_tickers = len(tickers)
    weights = np.array([1 / num_tickers] * num_tickers)  # Equal weights for all tickers
    
    # Combine individual returns into a DataFrame
    portfolio_returns = pd.concat(all_returns, axis=1)  # Equally weighted portfolio

    mean_returns = portfolio_returns.mean()
    
    cov_matrix = portfolio_returns.cov()
    
     # Call portfolio_annualised_performance to calculate annualized return and volatility
    portfolio_volatility, portfolio_annual_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    
        
    # Convert annual risk-free rate to daily
    rf_daily = rf_rate / 252

    # Calculate portfolio metrics
    portfolio_sharpe_ratio = np.mean(portfolio_returns.values - rf_daily) / portfolio_volatility
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    # Find the peak and minimum data points
    peak_value = portfolio_returns.values.max()
    #peak_date = portfolio_returns.values.idxmax()
    min_value = portfolio_returns.values.min()
    #min_date = portfolio_returns.values.idxmin()
    
    MDD = round(((peak_value-min_value)/peak_value) * 100,4)
    
    #portfolio_annual_return = (1 + portfolio_returns.mean()) ** 12 - 1
    portfolio_calmar_ratio = portfolio_annual_return / MDD if MDD != 0 else np.nan
    downside_returns = portfolio_returns[portfolio_returns < rf_daily]
    downside_deviation = np.std(downside_returns.values)
    portfolio_sortino_ratio = np.mean(portfolio_returns.values - rf_daily) / downside_deviation if downside_deviation != 0 else np.nan

    # Return portfolio metrics as a dictionary
    return {
        "Ticker": "Portfolio",
        "Alpha": "N/A",  # Alpha is not calculated for the portfolio
        "Beta": "N/A",  # Beta is not calculated for the portfolio
        "Volatility": f"{portfolio_volatility * 100:.2f}%",  # Format as percentage
        "Sharpe Ratio": portfolio_sharpe_ratio,  # Keep as a ratio
        "Calmar Ratio": portfolio_calmar_ratio,  # Keep as a ratio
        "Annualized Return": f"{portfolio_annual_return * 100:.2f}%",  # Format as percentage
        "Sortino Ratio": portfolio_sortino_ratio,  # Keep as a ratio
        "Expense Ratio": "N/A"  # Expense ratio is not meaningful for the portfolio
    }

# Example usage:
tickers = ['AAPL', 'MSFT', 'GOOGL']
all_returns = get_all_returns(tickers = ['AAPL', 'MSFT', 'GOOGL'], start='2024-01-01', end='2025-01-01')
portfolio_metrics = calculate_portfolio_metrics(
                    tickers = tickers,
                    all_returns=all_returns,
                    rf_rate=0.01  # Pass the risk-free rate
                )
print(portfolio_metrics)

