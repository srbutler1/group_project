# group_project
The WRDS Portfolio Optimization Tool is a comprehensive Python-based application designed to streamline portfolio analysis and optimization. Utilizing data from the WRDS database and financial libraries like PyPortfolioOpt, the tool enables users to fetch top-performing mutual funds, analyze their holdings, and optimize investment portfolios based on advanced financial strategies such as Sharpe Ratio maximization, Equal Weight allocation, and Risk Parity.

This tool provides a user-friendly interface powered by Streamlit, allowing users to interactively explore data, fetch tickers, and visualize efficient frontiers for optimal investment decisions.

File path should be: 
my_project/
├── wrds_tool/
│   ├── __init__.py  # from .wrds_tool import connect_to_wrds, get_top_500_funds, fetch_tickers, optimize_portfolio, plot_efficient_frontier
│   ├── wrds_tool.py
├── wrds_ui.py
