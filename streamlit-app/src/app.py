import streamlit as st
from utils.data_utils import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# Set custom page title and icon
st.set_page_config(page_title="Streamlit App", page_icon="📊")

def main():
    st.title("Portfolio v Securities - Return and Risk Monitor")
    st.sidebar.title("Navigation")
    
    # Add navigation options here
    page = st.sidebar.selectbox("Select a page", ["Home", "Explanation", "About"])
    
    if page == "Home":
        st.subheader("Individual Asset and Portfolio Metrics")
        
        # Input box for multiple tickers (comma-separated)
        tickers_input = st.text_input("Enter Ticker Symbols (comma-separated)", value='AAPL, MSFT, GOOGL') #"SPY,NLR"
        tickers = [ticker.strip() for ticker in tickers_input.split(",")]  # Split and clean input
        
        # Input box for start and end dates
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))
        
        # Input box for risk-free rate
        rf_rate = st.number_input("Risk-Free Rate (Annual)", value=0.01, step=0.001)
        
        # Button to trigger calculation
        if st.button("Calculate Security Level Metrics"):
            try:
                # Get returns for selected tickers and time periods
                all_returns = get_all_returns(
                    tickers = tickers, 
                    start=start_date.strftime("%Y-%m-%d"), 
                    end=end_date.strftime("%Y-%m-%d"))
                
                
                # Call the calculate_metrics function                            
                results_df = calculate_metrics(
                    all_returns,
                    tickers=tickers,
                    rf_rate=rf_rate)
                
                # Store the results in session state
                st.session_state['results_df'] = results_df
                #st.dataframe(results_df, use_container_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
                
                
        # Button to trigger calculation for portfolio-level metrics
        if st.button("Calculate Portfolio Level Metrics"):
            try:
                # Ensure all_returns is already calculated
                if 'all_returns' not in locals():
                    all_returns = get_all_returns(
                        tickers=tickers, 
                        start=start_date.strftime("%Y-%m-%d"), 
                        end=end_date.strftime("%Y-%m-%d")
                    )
                
                # Call the calculate_portfolio_metrics function
                portfolio_metrics = calculate_portfolio_metrics(
                    tickers = tickers,
                    all_returns=all_returns,
                    rf_rate=rf_rate  # Pass the risk-free rate
                )
                
                # Display the portfolio metrics in a session state
                st.session_state['portfolio_metrics'] = portfolio_metrics
              
            except Exception as e:
                st.error(f"An error occurred: {e}")
                
        # Display the results DataFrame if it exists in session state        
        if 'results_df' in st.session_state:
            st.write("### Security Level Metrics")
            st.dataframe(st.session_state['results_df'], use_container_width=True)

        # Display the portfolio metrics if they exist in session state
        if 'portfolio_metrics' in st.session_state:
            st.write("### Portfolio Level Metrics")
            st.dataframe(pd.DataFrame([st.session_state['portfolio_metrics']]), use_container_width=True)
            
            
            # Create and display the correlation plot
            try:
                # Ensure all_returns is already calculated
                if 'all_returns' not in locals():
                    all_returns = get_all_returns(
                        tickers=tickers, 
                        start=start_date.strftime("%Y-%m-%d"), 
                        end=end_date.strftime("%Y-%m-%d")
                    )
                
                # Create a DataFrame of all returns for correlation calculation
                returns_df = pd.concat(all_returns, axis=1)
                returns_df.columns = tickers  # Set column names to ticker symbols

                # Calculate the correlation matrix
                correlation_matrix = returns_df.corr()

                # Plot the correlation heatmap
                st.write("### Correlation Plot Among Selected Tickers")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
                
                # Calculate correlation matrix
                #correlation_matrix = calculate_correlation_matrix(returns_df)
                
                num_tickers = len(tickers)
                weights = np.array([1 / num_tickers] * num_tickers)  # Equal weights for all tickers

                # Calculate diversification factor
                pdf = calculate_diversification_factor(weights, correlation_matrix)
               # Display the diversification factor with a heading
                st.write("### Portfolio Diversification Factor")
                st.write(f"The Portfolio Diversification Factor (PDF) is: **{pdf:.2f}**")
                
                
            except Exception as e:
                st.error(f"An error occurred while generating the correlation plot: {e}")
       
    elif page == "Explanation":
        st.subheader("Understanding the results")
        st.write("""
        This page contains information about the results:

        - **Alpha**: Represents the excess return of the asset relative to the benchmark.
        - **Beta**: Measures the volatility of the asset relative to the benchmark.
        - **Volatility**: The standard deviation of the asset's returns.
        - **Sharpe Ratio**: The risk-adjusted return of the asset.
        - **Calmar Ratio**: The ratio of annualized return to maximum drawdown. Helps
                            investors understand how much return they are getting for each 
                            unit of drawdown risk they are taking.
                            Hedge Fund target: 1.5 - 2
                            Mutual Fund target: 1.0 - 1.5

        - **Annualized Return**: The compounded annual growth rate of the asset.
        - **Sortino Ratio**: The risk-adjusted return considering only downside risk.
        - **Expense Ratio**: The percentage of assets deducted annually for expenses.
        """)
        
    elif page == "About":
        st.subheader("About Page")
        st.write("This page contains information about the application.")
        
        


if __name__ == "__main__":
    main()