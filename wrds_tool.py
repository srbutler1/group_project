import wrds
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
import logging

logging.basicConfig(
    filename='debug_wrds_tool.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def connect_to_wrds(username):
    try:
        conn = wrds.Connection(wrds_username=username)
        logging.info("Connected to WRDS successfully.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to WRDS: {e}")
        raise

def get_top_500_funds(conn):
    query = """
        WITH returns_10y AS (
            SELECT 
                crsp_fundno,
                EXP(SUM(LOG(1 + mret))) - 1 AS cumulative_return
            FROM 
                crsp_q_mutualfunds.monthly_tna_ret_nav
            WHERE 
                caldt >= CURRENT_DATE - INTERVAL '10 years'
                AND mret IS NOT NULL
            GROUP BY 
                crsp_fundno
        )
        SELECT 
            crsp_fundno, cumulative_return
        FROM 
            returns_10y
        ORDER BY 
            cumulative_return DESC
        LIMIT 500;
    """
    try:
        result = conn.raw_sql(query)
        if result.empty:
            logging.warning("No results for top 500 funds.")
            return None
        return result['crsp_fundno'].tolist()
    except Exception as e:
        logging.error(f"Error retrieving funds: {e}")
        raise

def fetch_tickers(conn, query_type="holdings", top_funds=None, top_n=50):
    if not top_funds:
        raise ValueError("No top funds provided for ticker fetching.")
    try:
        fund_list = ','.join(f"'{fund}'" for fund in top_funds)  # Convert list to SQL-compatible string
        if query_type == "holdings":
            query = f"""
                SELECT 
                    hc.ticker,
                    SUM(h.market_val) AS total_mutual_fund_holding_value
                FROM 
                    crsp.holdings AS h
                JOIN 
                    crsp.holdings_co_info AS hc
                ON 
                    h.crsp_company_key = hc.crsp_company_key
                JOIN 
                    crsp_q_mutualfunds.portnomap AS pm
                ON 
                    h.crsp_portno = pm.crsp_portno
                WHERE 
                    h.report_dt = (SELECT MAX(report_dt) FROM crsp.holdings)
                    AND pm.crsp_fundno IN ({fund_list})
                GROUP BY 
                    hc.ticker
                ORDER BY 
                    total_mutual_fund_holding_value DESC
                LIMIT {top_n};
            """
        elif query_type == "bought":
            query = f"""
                SELECT 
                    h1.ticker,
                    h1.security_name,
                    (SUM(h2.market_val) - SUM(h1.market_val)) AS change_in_market_value
                FROM 
                    crsp.holdings AS h1
                JOIN 
                    crsp.holdings AS h2
                ON 
                    h1.ticker = h2.ticker
                    AND h1.security_name = h2.security_name
                JOIN 
                    crsp_q_mutualfunds.portnomap AS pm
                ON 
                    h1.crsp_portno = pm.crsp_portno
                    AND h2.crsp_portno = pm.crsp_portno
                WHERE 
                    h1.report_dt = (SELECT MAX(report_dt) FROM crsp.holdings) - INTERVAL '3 months'
                    AND h2.report_dt = (SELECT MAX(report_dt) FROM crsp.holdings)
                    AND pm.crsp_fundno IN ({fund_list})
                GROUP BY 
                    h1.ticker, h1.security_name
                ORDER BY 
                    change_in_market_value DESC
                LIMIT {top_n};
            """
        else:
            raise ValueError("Invalid query type. Use 'holdings' or 'bought'.")

        result = conn.raw_sql(query)
        if result.empty:
            logging.warning(f"No data returned for the {query_type} query.")
            return None, None
        return result['ticker'].tolist(), result

    except Exception as e:
        logging.error(f"Error fetching tickers: {e}")
        raise
def optimize_portfolio(tickers, rf, investment_amount, hist_data, strategy):
    try:
        # Fetch historical data
        start_date = (datetime.today() - timedelta(days=int(hist_data) * 365)).strftime('%Y-%m-%d')
        end_date = datetime.today().strftime('%Y-%m-%d')
        prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close'].dropna(axis=1)

        # Drop tickers with insufficient data
        dropped_tickers = set(tickers) - set(prices.columns)
        if dropped_tickers:
            print(f"Tickers with insufficient data dropped: {', '.join(dropped_tickers)}")

        if prices.empty:
            raise ValueError("No valid price data available for the given tickers.")

        # Calculate expected returns and covariance matrix
        mu = mean_historical_return(prices)
        S = sample_cov(prices)
        if S.isnull().values.any() or mu.isnull().values.any():
            raise ValueError("Covariance matrix or expected returns contain NaN values.")

        # Initialize EfficientFrontier
        ef = EfficientFrontier(mu, S)

        # Strategy-specific optimization
        if strategy == "E":
            weights = {ticker: 1 / len(prices.columns) for ticker in prices.columns}
        elif strategy == "S":
            ef.add_constraint(lambda w: w >= 0.01)
            ef.add_constraint(lambda w: w <= 0.15)
            weights = ef.max_sharpe(risk_free_rate=rf)
        elif strategy == "R":
            ef.add_objective(objective_functions.L2_reg, gamma=1)
            weights = ef.min_volatility()
        else:
            raise ValueError("Invalid strategy. Choose 'E', 'S', or 'R'.")

        # Clean weights for optimization-based strategies
        if strategy in ["S", "R"]:
            weights = ef.clean_weights()

        # Calculate allocations
        allocations = {ticker: weight * investment_amount for ticker, weight in weights.items() if weight > 0}

        # Portfolio performance
        performance = ef.portfolio_performance(risk_free_rate=rf) if strategy in ["S", "R"] else (None, None, None)

        return allocations, {"mu": mu, "S": S, "ef": ef, "performance": performance}
    except Exception as e:
        print(f"Error during portfolio optimization: {e}")
        return None, None


import matplotlib.pyplot as plt
import numpy as np

def plot_efficient_frontier(mu, S, rf, ef):
    """
    Plots the efficient frontier and the optimized portfolio.

    Parameters:
        mu (pd.Series): Expected returns for each asset.
        S (pd.DataFrame): Covariance matrix of asset returns.
        rf (float): Risk-free rate.
        ef (EfficientFrontier): PyPortfolioOpt EfficientFrontier object.

    Returns:
        Matplotlib figure: For embedding in Streamlit.
    """
    try:
        # Validate input data
        if mu.empty or S.empty:
            raise ValueError("Expected returns or covariance matrix is empty.")

        # Generate random portfolios for visualization
        n_samples = min(5000, len(mu) * 100)  # Adjust number of samples based on available tickers
        np.random.seed(42)
        w = np.random.dirichlet(np.ones(len(mu)), size=n_samples)
        rets = w @ mu.values
        stds = np.sqrt(np.diag(w @ S.values @ w.T))
        sharpe_ratios = (rets - rf) / stds

        # Validate generated data
        if len(rets) == 0 or len(stds) == 0:
            raise ValueError("Generated returns or volatilities are empty.")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(stds, rets, c=sharpe_ratios, cmap="viridis", alpha=0.6)
        plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")
        ax.set_xlabel("Volatility (Standard Deviation)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Efficient Frontier with Capital Market Line")

        # Plot the optimized portfolio
        optimized_return, optimized_volatility, optimized_sharpe = ef.portfolio_performance(risk_free_rate=rf)
        ax.scatter(optimized_volatility, optimized_return, c="red", s=100, label="Optimized Portfolio")

        # Plot the capital market line (CML)
        x = np.linspace(0, max(stds), 100)
        y = rf + optimized_sharpe * x
        ax.plot(x, y, color="orange", label="Capital Market Line (CML)")

        ax.legend()
        ax.grid()

        return fig  # Return figure for Streamlit
    except Exception as e:
        print(f"Error plotting efficient frontier: {e}")


    
