import streamlit as st
from wrds_tool import connect_to_wrds, get_top_500_funds, fetch_tickers, optimize_portfolio, plot_efficient_frontier

def main():
    st.title("WRDS Portfolio Optimization Tool")

    # Initialize session state for variables
    if "conn" not in st.session_state:
        st.session_state.conn = None
    if "top_funds" not in st.session_state:
        st.session_state.top_funds = None
    if "tickers" not in st.session_state:
        st.session_state.tickers = None

    # WRDS Connection
    username = st.text_input("Enter your WRDS username:", key="username")
    if st.button("Connect to WRDS"):
        try:
            st.session_state.conn = connect_to_wrds(username)
            st.success("Connected to WRDS!")
        except Exception as e:
            st.error(f"Failed to connect to WRDS: {e}")

    # Fetch Top 500 Funds
    if st.session_state.conn:
        if st.button("Fetch Top 500 Funds"):
            try:
                st.session_state.top_funds = get_top_500_funds(st.session_state.conn)
                if st.session_state.top_funds:
                    st.success("Top 500 funds retrieved successfully!")
                    st.write(st.session_state.top_funds[:10])  # Display top 10 for preview
                else:
                    st.error("No funds retrieved.")
            except Exception as e:
                st.error(f"Failed to fetch funds: {e}")

    # Fetch Tickers
    if st.session_state.top_funds:
        query_type = st.selectbox("Query Type", ["holdings", "bought"])
        top_n = st.slider("Number of Tickers to Fetch", min_value=10, max_value=50, step=10)
        if st.button("Fetch Tickers"):
            try:
                tickers, result = fetch_tickers(st.session_state.conn, query_type, st.session_state.top_funds, top_n=top_n)
                st.session_state.tickers = tickers
                if tickers:
                    st.success("Tickers retrieved successfully!")
                    st.write(tickers)  # Display tickers
                else:
                    st.error("No tickers retrieved.")
            except Exception as e:
                st.error(f"Failed to fetch tickers: {e}")

    # Portfolio Optimization
    if st.session_state.tickers:
        st.header("Portfolio Optimization")
        rf = st.number_input("Risk-Free Rate (e.g., 0.03)", min_value=0.0, step=0.01)
        investment_amount = st.number_input("Investment Amount (USD)", min_value=1000.0, step=100.0)
        hist_data = st.selectbox("Historical Data Range (Years)", [1, 3, 5, 10])
        strategy = st.selectbox("Optimization Strategy", ["E (Equal Weight)", "S (Sharpe Ratio)", "R (Risk Parity)"])

    if st.button("Optimize Portfolio"):
        try:
            allocations, performance_data = optimize_portfolio(
                st.session_state.tickers, rf, investment_amount, hist_data, strategy[0]
            )
            if allocations:
                st.success("Portfolio optimized successfully!")
                st.write("Allocations:", allocations)
                st.write("Performance Metrics:", performance_data["performance"])

                # Debug: Show remaining tickers
                st.write("Remaining Tickers:", list(performance_data['mu'].index))

                # Plot Efficient Frontier for optimization strategies
                if strategy[0] in ["S", "R"]:
                    st.subheader("Efficient Frontier")
                    fig = plot_efficient_frontier(
                        mu=performance_data['mu'], 
                        S=performance_data['S'], 
                        rf=rf, 
                        ef=performance_data['ef']
                    )
                    st.pyplot(fig)
            else:
                st.error("Failed to optimize portfolio.")
        except Exception as e:
            st.error(f"Portfolio optimization error: {e}")

if __name__ == "__main__":
    main()

