import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="Autonomous - Strategy Performance & Risk Dashboard")

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

    downside_vols = []
    win_rates = []
    avg_wins = []
    avg_losses = []
    profit_factors = []

    for col in returns.columns:
        col_returns = returns[col]

        cumulative = (1 + col_returns).cumprod()
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

        # Downside volatility
        downside_returns = col_returns[col_returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else 0
        downside_vol = downside_std * np.sqrt(365)
        downside_vols.append(downside_vol)

        # Win Rate: % of positive return days over whole period
        positive = col_returns > 0
        win_rate = positive.mean() if len(col_returns) > 0 else 0
        win_rates.append(win_rate)

        # Average Daily Profit
        avg_win = col_returns[positive].mean() if positive.any() else 0
        avg_wins.append(avg_win)

        # Lose Rate (internal for profit factor): % of negative return days over whole period
        negative = col_returns < 0
        lose_rate = negative.mean() if len(col_returns) > 0 else 0

        # Average Daily Loss (positive magnitude)
        avg_loss_mag = -col_returns[negative].mean() if negative.any() else 0  # negative mean, so - to make positive
        avg_losses.append(avg_loss_mag)

        # Profit Factor
        if lose_rate > 0 and avg_loss_mag > 0:
            profit_factor = (win_rate * avg_win) / (lose_rate * avg_loss_mag)
        else:
            profit_factor = np.inf if win_rate > 0 else 0
        profit_factors.append(profit_factor)

    max_drawdown = pd.Series(max_drawdowns, index=returns.columns)
    longest_drawdown_length = pd.Series(longest_drawdowns, index=returns.columns)
    downside_vol = pd.Series(downside_vols, index=returns.columns)
    win_rate_series = pd.Series(win_rates, index=returns.columns)
    avg_win_series = pd.Series(avg_wins, index=returns.columns)
    avg_loss_series = pd.Series(avg_losses, index=returns.columns)
    profit_factor_series = pd.Series(profit_factors, index=returns.columns)

    sharpe_ratio = annualized_return / annualized_volatility
    sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], 0).fillna(0)

    calmar_ratio = annualized_return / max_drawdown.abs()
    calmar_ratio = calmar_ratio.replace([np.inf, -np.inf], 0).fillna(0)

    sortino_ratio = annualized_return / downside_vol
    sortino_ratio = sortino_ratio.replace([np.inf, -np.inf], 0).fillna(0)

    # First part of metrics
    metrics_df1 = pd.DataFrame({
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Max Drawdown": max_drawdown,
        "Longest Drawdown (days)": longest_drawdown_length,
        "Sharpe Ratio": sharpe_ratio,
        "Calmar Ratio": calmar_ratio,
        "Sortino Ratio": sortino_ratio,
    })

    # Second part of metrics (removed Lose Rate)
    metrics_df2 = pd.DataFrame({
        "Win Rate": win_rate_series,
        "Average Daily Profit": avg_win_series,
        "Average Daily Loss": avg_loss_series,
        "Profit Factor": profit_factor_series,
    })

    # Transpose both
    metrics_df1 = metrics_df1.T
    metrics_df2 = metrics_df2.T

    # Create blank row with NaN
    blank_df = pd.DataFrame(index=[""], columns=metrics_df1.columns)

    # Concatenate: first part, blank, second part
    metrics_df = pd.concat([metrics_df1, blank_df, metrics_df2])

    metrics_df = metrics_df.round(4)
    metrics_df = metrics_df.fillna("")  # Replace NaN with empty string for blank row

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

    # Define red font function
    def red_font_if_negative(val):
        try:
            return 'color: red' if float(val) < 0 else ''
        except (ValueError, TypeError):
            return ''

    # Format dict
    format_dict = {
        "Cumulative Return": "{:.2%}",
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Max Drawdown": "{:.2%}",
        "Longest Drawdown (days)": "{:.0f}",
        "Sharpe Ratio": "{:.2f}",
        "Calmar Ratio": "{:.2f}",
        "Sortino Ratio": "{:.2f}",
        "Win Rate": "{:.2%}",
        "Average Daily Profit": "{:.2%}",
        "Average Daily Loss": "{:.2%}",
        "Profit Factor": "{:.2f}"
    }
    
    # Part 1: Key Metrics
    st.subheader("1. Key Metrics")
    metrics_df = calculate_metrics(data_selected)
    styled = metrics_df.style
    for metric, fmt in format_dict.items():
        styled = styled.format(fmt, subset=pd.IndexSlice[metric, :])
    styled = styled.map(red_font_if_negative)
    st.dataframe(styled)

    # Part 2: Cumulative Return chart
    st.subheader("2. Cumulative Return Chart")
    cum_fig = plot_cumulative_returns(data_selected)
    st.plotly_chart(cum_fig, use_container_width=True)

    # Part 3: Cumulative Return by Year table
    cum_return_year_df = cumulative_return_by_year(data_selected)
    
    # Sort by year ascending
    cum_return_year_df = cum_return_year_df.sort_index(ascending=True)
    
    styled_yearly = cum_return_year_df.style.map(red_font_if_negative).format("{:.2%}")
    
    st.subheader("3. Cumulative Return by Year")
    st.dataframe(styled_yearly)

    # Part 4: Correlation matrix heatmap for more than 2 strategies
    if len(selected_strategies) > 2:
        st.subheader("4. Correlation Matrix Heatmap")
        corr_fig = plot_correlation_matrix(data_selected)
        st.plotly_chart(corr_fig, use_container_width=True)

    # Part 5: Strategy Combination
    st.subheader("5. Strategy Combination")

    if selected_strategies:
        col1, col2 = st.columns([2, 3])
        with col1:
            weights = {}
            total = 0.0
            # Header
            h1, h2 = st.columns([3,1])
            h1.write("Strategy")
            h2.write("Weights (%)")
            for strat in selected_strategies:
                c1, c2 = st.columns([3,1])
                c1.write(strat)
                initial_value = 100.0 / len(selected_strategies)
                weight = c2.number_input(
                    "Weight",
                    key=f"weight_{strat}",
                    min_value=0.0,
                    value=initial_value,
                    step=0.01,
                    format="%.2f",
                    label_visibility="collapsed"
                )
                weights[strat] = weight
                total += weight
            # Sum row
            s1, s2 = st.columns([3,1])
            s1.write("Total")
            s2.write(f"{total:.2f}")

        with col2:
            run_button = st.button("Run")

        if run_button:
            if total == 0:
                st.warning("Sum of weights is 0.")
            else:
                combined_returns = pd.Series(0, index=data_selected.index)
                for strat, weight in weights.items():
                    combined_returns += (weight / 100.0) * data_selected[strat]
                combined_df = pd.DataFrame({"Combined": combined_returns})
                metrics_combined = calculate_metrics(combined_df)
                cbstyled = metrics_combined.style
                for metric, fmt in format_dict.items():
                    cbstyled = cbstyled.format(fmt, subset=pd.IndexSlice[metric, :])
                cbstyled = cbstyled.map(red_font_if_negative)
                
                cum_fig_combined = plot_cumulative_returns(combined_df)

                output_col1, output_col2 = st.columns(2)
                with output_col1:
                    st.subheader("Combined Cumulative Return Chart")
                    st.plotly_chart(cum_fig_combined, use_container_width=True)
                with output_col2:
                    st.subheader("Combined Strategy Metrics")
                    st.dataframe(cbstyled, use_container_width=True)
    else:
        st.info("Select strategies to combine.")

#if __name__ == "__main__":
main()

