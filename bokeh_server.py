import pandas as pd
import numpy as np
import math
from bokeh.models import (
    ColumnDataSource,
    CrosshairTool,
    HoverTool,
    NumeralTickFormatter,
)
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.transform import cumsum
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import sys
import os
import socket

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=5006, max_attempts=10):
    """Find an available port starting from start_port"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def make_portfolio_app(doc):
    # Define the number of portfolios to simulate
    num_port = 5000

    # Load data
    if getattr(sys, "frozen", False):
        # Running in a bundle
        filePath = os.path.join(sys._MEIPASS, "myport2.csv")
    else:
        # Running in a normal Python environment
        filePath = "./myport2.csv"

    df = pd.read_csv(filePath)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.set_index(["Date"], inplace=True)

    # Ensure all columns are numeric
    price_data = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN values
    df_clean = price_data.dropna()

    # Calculate log returns
    log_ret = np.log(df_clean / df_clean.shift(1))

    # Calculate covariance matrix of log returns
    cov_mat = log_ret.cov() * 252

    # Initialize arrays for portfolio weights, returns, risk, and Sharpe ratios
    all_wts = np.zeros((num_port, len(df_clean.columns)))
    port_returns = np.zeros(num_port)
    port_risk = np.zeros(num_port)
    sharpe_ratio = np.zeros(num_port)

    # Simulate random portfolios
    np.random.seed(42)
    for i in range(num_port):
        # Generate random portfolio weights
        wts = np.random.uniform(size=len(df_clean.columns))
        wts = wts / np.sum(wts)
        all_wts[i, :] = wts

        # Calculate portfolio return
        port_ret = np.sum(log_ret.mean() * wts)
        port_ret = (port_ret + 1) ** 252 - 1
        port_returns[i] = port_ret

        # Calculate portfolio risk (standard deviation)
        port_sd = np.sqrt(np.dot(wts.T, np.dot(cov_mat, wts)))
        port_risk[i] = port_sd

        # Calculate Sharpe Ratio, assuming a risk-free rate of 0%
        sr = port_ret / port_sd
        sharpe_ratio[i] = sr

    # Identify portfolios with max Sharpe ratio, max return, and minimum variance
    max_sr_idx = sharpe_ratio.argmax()
    max_ret_idx = port_returns.argmax()
    min_var_idx = port_risk.argmin()

    max_sr_ret = port_returns[max_sr_idx]
    max_sr_risk = port_risk[max_sr_idx]
    max_sr_w = all_wts[max_sr_idx, :]

    max_ret_ret = port_returns[max_ret_idx]
    max_ret_risk = port_risk[max_ret_idx]
    max_ret_w = all_wts[max_ret_idx, :]

    min_var_ret = port_returns[min_var_idx]
    min_var_risk = port_risk[min_var_idx]
    min_var_w = all_wts[min_var_idx, :]

    # Efficient frontier plot
    p = figure(
        height=700,
        width=770,
        title=f"Efficient Frontier. Simulations: {num_port}",
        tools="box_zoom,wheel_zoom,reset",
        toolbar_location="above",
    )
    p.add_tools(CrosshairTool(line_alpha=1, line_color="lightgray", line_width=1))
    p.add_tools(HoverTool(tooltips=None))
    source = ColumnDataSource(data=dict(risk=port_risk, profit=port_returns))
    p.circle(
        x="risk",
        y="profit",
        source=source,
        line_alpha=0,
        hover_color="navy",
        alpha=0.4,
        hover_alpha=1,
        size=8,
    )
    p.circle(
        min_var_risk,
        min_var_ret,
        color="tomato",
        legend_label="Portfolio with minimum variance",
        size=10,
    )
    p.circle(
        max_sr_risk,
        max_sr_ret,
        color="orangered",
        legend_label="Portfolio with max Sharpe ratio",
        size=12,
    )
    p.circle(
        max_ret_risk,
        max_ret_ret,
        color="firebrick",
        legend_label="Portfolio with max return",
        size=9,
    )
    p.legend.location = "top_left"
    p.xaxis.axis_label = "Volatility, or risk (standard deviation)"
    p.yaxis.axis_label = "Annual return"
    p.xaxis[0].formatter = NumeralTickFormatter(format="0.0%")
    p.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")

    # Portfolio composition plots
    def plot_portfolio_composition(ticks, weights, plot_name):
        x = dict()
        for i in range(len(ticks)):
            x[ticks[i]] = weights[i]

        color_list = [
            "olive",
            "yellowgreen",
            "lime",
            "chartreuse",
            "springgreen",
            "lightgreen",
            "darkseagreen",
            "seagreen",
            "green",
            "darkgreen",
        ]

        plot_data = (
            pd.Series(x).reset_index(name="value").rename(columns={"index": "stock"})
        )
        plot_data["angle"] = plot_data["value"] / plot_data["value"].sum() * 2 * math.pi
        plot_data["color"] = color_list[: len(weights)]
        p = figure(
            height=250,
            width=250,
            title=plot_name,
            toolbar_location=None,
            tools="hover",
            tooltips="@stock: @value{%0.1f}",
            x_range=(-0.5, 1.0),
        )
        p.wedge(
            x=0,
            y=1,
            radius=0.4,
            start_angle=cumsum("angle", include_zero=True),
            end_angle=cumsum("angle"),
            line_color="white",
            color="color",
            source=plot_data,
        )
        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None
        p.outline_line_color = None

        return p

    ticks = df_clean.columns
    p_minvar = plot_portfolio_composition(
        ticks, min_var_w, "Portfolio with minimum variance"
    )
    p_maxsr = plot_portfolio_composition(ticks, max_sr_w, "Portfolio with max Sharpe ratio")
    p_maxret = plot_portfolio_composition(ticks, max_ret_w, "Portfolio with max return")

    # Time series of stock prices over time
    start_date = df_clean.index.min().strftime("%d/%m/%Y")
    end_date = df_clean.index.max().strftime("%d/%m/%Y")
    p_time = figure(
        height=450,
        width=675,
        toolbar_location=None,
        tools="",
        title=f"Time series of stock prices in time. From {start_date} to {end_date}",
    )
    color_list = [
        "olive",
        "yellowgreen",
        "lime",
        "chartreuse",
        "springgreen",
        "lightgreen",
        "darkseagreen",
        "seagreen",
        "green",
        "darkgreen",
    ]

    for i, tick in enumerate(ticks):
        p_time.line(
            df_clean.index,
            df_clean[tick] / df_clean[tick].iloc[0],
            color=color_list[i % len(color_list)],
            line_width=1,
            legend_label=tick,
        )

    # Adding portfolio with minimum Sharpe ratio
    val_max_shr = np.dot(df_clean / df_clean.iloc[0], max_sr_w)
    p_time.line(
        df_clean.index,
        val_max_shr,
        legend_label="Portfolio with min SR",
        color="orangered",
        line_width=2.5,
    )

    p_time.legend.location = "top_left"
    p_time.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
    p_time.xaxis.axis_label = "Trading day"
    p_time.yaxis.axis_label = "Return"

    # Create dashboard layout
    layout = row([p, column([p_time, row([p_minvar, p_maxsr, p_maxret])])])
    
    # Add the layout to the document
    doc.add_root(layout)
    doc.title = "Portfolio Optimization Dashboard"

def main():
    # Create a Bokeh application
    app = Application(FunctionHandler(make_portfolio_app))
    
    # Find an available port
    port = find_available_port()
    
    # For development purposes only - allow all origins
    # In production, you would specify exact origins for security
    import os
    os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "*"
    
    # Create and configure a server
    server = Server(
        {'/': app}, 
        num_procs=1, 
        port=port, 
        address='localhost'
    )
    
    # Start the server
    server.start()
    
    print(f"Bokeh server running at http://localhost:{port}")
    print("To view in browser, navigate to the URL above")
    print("Press Ctrl+C to stop the server")
    
    # Keep the server running
    try:
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
    except KeyboardInterrupt:
        print("Server shut down")

if __name__ == "__main__":
    main()
