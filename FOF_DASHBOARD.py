import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Strategy Returns Dashboard")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=["Date"], dayfirst=True)
    df = df.set_index("Date").sort_index()
    return df

def calculate_metrics(df_selected):
    returns = df_selected

    cumulative_return = (1 + returns).prod() - 1

    total_days = returns.shape[0]
    years = total_days / 365

    annualized_return = (1 + cumulative_return) ** (1 / years) - 1

    annualized_volatility = returns.std() * np.sqrt(365)

    max_drawdowns = []
    longest_drawdowns = []

    for col in returns.columns:
        cumulative = (1 + returns[col]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_drawdown = drawdown.min()
        max_drawdowns.append(max_drawdown)

        # Calculate longest drawdown period
        dd = drawdown < 0
        max_len = 0
        current_len = 0
        for val in dd:
            if val:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        longest_drawdowns.append(max_len)

    max_drawdown = pd.Series(max_drawdowns, index=returns.columns)
    longest_drawdown_length = pd.Series(longest_drawdowns, index=returns.columns)

    sharpe_ratio = annualized_return / annualized_volatility

    calmar_ratio = annualized_return / max_drawdown.abs()

    metrics_df = pd.DataFrame({
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Max Drawdown": max_drawdown,
        "Longest Drawdown (days)": longest_drawdown_length,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio,
    })

    metrics_df = metrics_df.round(4)

    return metrics_df


def plot_cumulative_returns(df_selected):
    cumulative = (1 + df_selected).cumprod()
    fig = go.Figure()
    for col in cumulative.columns:
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[col],
                                 mode='lines', name=col))
    fig.update_layout(
        title="Cumulative Return of $1 Invested",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    return fig

def plot_correlation_matrix(df_selected):
    corr = df_selected.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=px.colors.diverging.RdYlGn_r,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Matrix (Daily Returns)"
    )
    fig.update_traces(textfont_size=16)  # increase number for larger font
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40), height=400)
    return fig

def cumulative_return_by_year(df_selected):
    df = df_selected.copy()
    df.index=pd.to_datetime(df.index)
    df["Year"] = df.index.year
    cum_returns_yearly = {}
    for year in df["Year"].unique():
        df_year = df[df["Year"] == year].drop(columns=["Year"])
        cum_return = (1 + df_year).prod() - 1
        cum_returns_yearly[year] = cum_return

    cum_returns_df = pd.DataFrame(cum_returns_yearly).T
    cum_returns_df.index.name = "Year"
    cum_returns_df = cum_returns_df.round(4)
    return cum_returns_df

def main():
    st.title("Strategy Returns Dashboard")

    uploaded_file = st.file_uploader("Upload CSV with Date and strategy daily returns", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to proceed.")
        return

    data = load_data(uploaded_file)

    # Strategy selector (multiselect)
    strategies = list(data.columns)
    selected_strategies = st.multiselect("Select Strategies to Analyze", strategies, default=strategies[:3])

    if not selected_strategies:
        st.warning("Please select at least one strategy.")
        return

    data_selected = data[selected_strategies].dropna()

    # Layout: Use columns to organize the four parts nicely
    # Part 1: Metrics dashboard at top
    
    format_dict = {
    "Cumulative Return": "{:.2%}",
    "Annualized Return": "{:.2%}",
    "Annualized Volatility": "{:.2%}",
    "Max Drawdown": "{:.2%}",
    "Longest Drawdown (days)": "{:.0f}",  # integer format
    "Sharpe Ratio": "{:.2f}",  # 2 decimal
    "Calmar Ratio": "{:.2f}"   # 2 decimal
    }
    
    st.subheader("1. Key Metrics")
    metrics_df = calculate_metrics(data_selected)
    st.dataframe(metrics_df.style.format(format_dict))

    # Part 2: Cumulative Return chart
    st.subheader("2. Cumulative Return Chart")
    cum_fig = plot_cumulative_returns(data_selected)
    st.plotly_chart(cum_fig, use_container_width=True)


    # Part 4: Cumulative Return by Year table
    cum_return_year_df = cumulative_return_by_year(data_selected)
    
    # Sort by year ascending
    cum_return_year_df = cum_return_year_df.sort_index(ascending=True)
    
    def red_font_if_negative(val):
        try:
            return 'color: red' if val < 0 else ''  # red font if negative
        except:
            return ''  # in case of non-numeric cells
    
    styled_yearly = cum_return_year_df.style.format("{:.2%}").map(red_font_if_negative)
    
    st.subheader("3. Cumulative Return by Year")
    st.dataframe(styled_yearly)

    # Part 3: Correlation matrix heatmap for more than 2 strategies
    if len(selected_strategies) > 2:
        st.subheader("4. Correlation Matrix Heatmap")
        corr_fig = plot_correlation_matrix(data_selected)
        st.plotly_chart(corr_fig, use_container_width=True)


#if __name__ == "__main__":
main()