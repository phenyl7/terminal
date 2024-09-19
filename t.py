import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import webbrowser 
from edgar import *

# File to save portfolio data
PORTFOLIO_FILE = 'portfolio.json'

# Load portfolio from JSON file
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save portfolio to JSON file
def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, 'w') as file:
        json.dump(portfolio, file, indent=4)

portfolio = load_portfolio()

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = {}
    
    hist_1y = stock.history(period="1y")
    
    if not hist_1y.empty:
        data['price_current'] = hist_1y['Close'].iloc[-1]  # Last close price
    else:
        data['price_current'] = None

    return data



def plot():
    import matplotlib.pyplot as plt
    import yfinance as yf
    from datetime import datetime

    def plot_stock_price():
        # Specify ticker symbol
        ticker = input("Enter the stock ticker: ")

        # Set the date range for YTD
        end_date = datetime.now()
        start_date = datetime(end_date.year, 1, 1)  # January 1st of the current year
        
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)

        if stock_data.empty:
            print(f"No data found for ticker: {ticker}")
            return

        # Calculate latest price and percent change
        latest_price = stock_data['Adj Close'].iloc[-1]
        initial_price = stock_data['Adj Close'].iloc[0]
        percent_change = ((latest_price - initial_price) / initial_price) * 100

        # Create subplots with specific height ratios
        fig, ax1 = plt.subplots(figsize=(12, 8), nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        
        # Set figure background color
        fig.patch.set_facecolor('black')

        # Plot stock price on the primary y-axis
        ax1[0].plot(stock_data.index, stock_data['Adj Close'], label=f'{ticker} Price', color='white')
        ax1[0].fill_between(stock_data.index, stock_data['Adj Close'], color='darkblue', alpha=0.4)

        # Add a text box with an orange background for latest price
        textstr = f'Latest Price: ${latest_price:.2f}\nChange: {percent_change:.2f}%'
        props = dict(boxstyle='round', facecolor='orange', alpha=1)
        ax1[0].text(0.05, 0.95, textstr, transform=ax1[0].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, color='black')

        # Adjust y-axis limits
        y_min = stock_data['Adj Close'].min()
        y_max = stock_data['Adj Close'].max()
        ax1[0].set_ylim(y_min * 0.90, y_max * 1.10)

        ax1[0].set_title(f'{ticker} YTD', color='white')
        ax1[0].set_ylabel('Price', color='white')
        ax1[0].legend()
        ax1[0].set_facecolor('black')  # Background color of the plot
        ax1[0].tick_params(axis='both', colors='orange')  # Color of the ticks
        ax1[0].grid(color='orange', linestyle='--', linewidth=0.5)  # Grid color and style

        # Plot volume bars on the secondary y-axis
        ax1[1].bar(stock_data.index, stock_data['Volume'], color='#536878', alpha=1)
        ax1[1].set_xlabel('Date', color='white')
        ax1[1].set_ylabel('Volume', color='white')
        ax1[1].set_facecolor('black')  # Background color of the plot
        ax1[1].tick_params(axis='both', colors='orange')  # Color of the ticks
        ax1[1].grid(color='orange', linestyle='--', linewidth=0.5)  # Grid color and style

        plt.tight_layout()  # Adjust layout to fit both plots
        plt.show(block=False)

    if __name__ == "__main__":
        plot_stock_price()



def port():
    import json

    def load_portfolio(filename='portfolio.json'):
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File {filename} not found. Initializing with empty portfolio.")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding {filename}. Initializing with empty portfolio.")
            return {}

    def save_portfolio(portfolio, filename='portfolio.json'):
        with open(filename, 'w') as file:
            json.dump(portfolio, file, indent=4)

    def main():
        global portfolio
        portfolio = load_portfolio()
        
        # Prompt the user for the number of years to plot
        try:
            years = float(input("Enter the number of years to plot: "))
            plot_portfolio_performance_chart(years)
        except ValueError:
            print("Invalid input. Please enter a numeric value for years.")

    def plot_portfolio_performance_chart(years):
        import yfinance as yf
        import numpy as np
        import matplotlib.pyplot as plt
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=float(years * 365))  # Use float to handle fractional days
        
        # Filter out ticker "1"
        tickers_to_plot = [ticker for ticker in portfolio if ticker != "1"]
        total_portfolio_value = 0
        asset_values = []
        stock_labels = []
        percent_changes = []
        
        # Iterate over each stock in the portfolio
        for ticker in tickers_to_plot:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if stock_data.empty:
                    print(f"No data found for ticker: {ticker}")
                    continue

                # Calculate the stock's total value in the portfolio
                last_close_price = stock_data['Close'].iloc[-1]
                asset_value = portfolio[ticker]['shares'] * last_close_price
                asset_values.append(asset_value)
                stock_labels.append(ticker)
                total_portfolio_value += asset_value
                
                # Calculate percent change over the entered years
                percent_change = ((last_close_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
                percent_changes.append(percent_change)
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        
        # Normalize asset values for the pie chart
        asset_values = np.array(asset_values)
        asset_percentages = (asset_values / total_portfolio_value * 100).round()  # Round to nearest whole number
        
        # Create the pie chart for asset allocation
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(tickers_to_plot)))  # Generate a colormap
        wedges, texts = axs[0].pie(asset_percentages, labels=None, startangle=90, colors=colors)
        
        axs[0].set_title('Portfolio Asset Allocation', color='white')
        
        # Format legend with tickers and their percentages
        legend_labels = [f"{ticker}: {percent:.0f}%" for ticker, percent in zip(stock_labels, asset_percentages)]
        axs[0].legend(wedges, legend_labels, title="Stocks", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10, title_fontsize=12)
        
        # Plot the bar chart for percent change over the period
        bars = axs[1].bar(stock_labels, percent_changes, color=colors, edgecolor='white')
        axs[1].set_title(f'Portfolio Performance Over {years:.1f} Year(s)', color='white')
        axs[1].set_xlabel('Stock Ticker', color='orange')
        axs[1].set_ylabel('%', color='orange')

        # Add data labels above bars
        for bar, percent in zip(bars, percent_changes):
            height = bar.get_height()
            axs[1].text(bar.get_x() + bar.get_width() / 2, height, f'{percent:.2f}%', ha='center', va='bottom', color='white')
        
        # Set axes and grid lines to orange
        axs[1].tick_params(axis='x', colors='orange')
        axs[1].tick_params(axis='y', colors='orange')
        axs[1].grid(color='orange', linestyle='--', linewidth=0.5)

        # Set background and tick colors for both plots
        for ax in axs:
            ax.set_facecolor('black')
        
        plt.tight_layout()
        plt.show(block=False)
        
        # Prompt to add or remove a position or do nothing
        while True:
            action = input("Do you want to (A)dd a position, (R)emove a position, or (N)othing? ").strip().upper()
            if action == 'A':
                add_position()
            elif action == 'R':
                remove_position()
            elif action == 'N':
                print("No changes made to the portfolio.")
                break
            else:
                print("Invalid option. Please choose (A)dd, (R)emove, or (N)othing.")

    def add_position():
        ticker = input("Enter ticker: ")
        cost_basis = float(input("Enter cost basis: "))
        shares = float(input("Enter number of shares: "))
        portfolio[ticker] = {'shares': shares, 'cost_basis': cost_basis}
        save_portfolio(portfolio)
        print(f"Added {ticker} to portfolio.")

    def remove_position():
        ticker = input("Enter ticker to remove: ")
        if ticker in portfolio:
            del portfolio[ticker]
            save_portfolio(portfolio)
            print(f"Removed {ticker} from portfolio.")
        else:
            print("Ticker not found in portfolio.")

    def save_portfolio(portfolio, filename='portfolio.json'):
        with open(filename, 'w') as file:
            json.dump(portfolio, file, indent=4)

    if __name__ == "__main__":
        main()



def gm():
    import json
    import yfinance as yf
    from datetime import datetime
    import webbrowser

    # Define ANSI color codes
    YELLOW = '\033[94m'  # Blue
    LIME_GREEN = '\033[92m'
    NEON_RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

    def get_greeting():
        now = datetime.now()
        hour = now.hour

        if 5 <= hour < 12:
            return "Good morning!"
        elif 12 <= hour < 17:
            return "Good afternoon!"
        else:
            return "Good evening!"

    def load_portfolio(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)

    def get_stock_data(ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m')
        
        # Latest price
        latest_price = data['Close'].iloc[-1] if not data.empty else None
        
        # Percent change since market open
        if len(data) > 1:
            opening_price = data['Open'].iloc[0]
            percent_change_today = ((latest_price - opening_price) / opening_price) * 100
        else:
            percent_change_today = None
        
        return latest_price, percent_change_today

    def calculate_percent_change(cost_basis, current_price):
        return ((current_price - cost_basis) / cost_basis) * 100

    def calculate_performance(portfolio):
        performance = {}
        total_value = 0
        for ticker, info in portfolio.items():
            if ticker == "1":
                continue
            
            latest_price, percent_change_today = get_stock_data(ticker)
            if latest_price is not None:
                percent_change = calculate_percent_change(info['cost_basis'], latest_price)
                value = info['shares'] * latest_price
                total_value += value
                
                performance[ticker] = {
                    'shares': info['shares'],
                    'cost_basis': info['cost_basis'],
                    'current_price': latest_price,
                    'percent_change': percent_change,
                    'percent_change_today': percent_change_today,
                    'value': value
                }
            else:
                performance[ticker] = {
                    'shares': info['shares'],
                    'cost_basis': info['cost_basis'],
                    'current_price': 'N/A',
                    'percent_change': 'N/A',
                    'percent_change_today': 'N/A',
                    'value': 'N/A'
                }
        
        return performance, total_value

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def print_performance(performance, total_value):
        for ticker, data in performance.items():
            print(f"{YELLOW}Ticker: {ticker.upper()}{RESET}")
            print(f"Shares: {data['shares']}")
            print(f"Cost Basis: ${data['cost_basis']:.2f}")
            print(f"Current Price: ${data['current_price']:.2f}" if data['current_price'] != 'N/A' else "Current Price: N/A")
            print(f"Δ: {colorize_percent(data['percent_change'])}" if data['percent_change'] != 'N/A' else "Percent Change: N/A")
            print(f"1D %Δ: {colorize_percent(data['percent_change_today'])}" if data['percent_change_today'] != 'N/A' else "Percent Change Today: N/A")
            print(f"Value: ${data['value']:.2f}" if data['value'] != 'N/A' else "Value: N/A")
            print()

        print(f"Total Portfolio Value: ${total_value:.2f}")

    # Main function logic
    print(get_greeting())

    portfolio_file = 'portfolio.json'  # Path to your portfolio.json file
    portfolio = load_portfolio(portfolio_file)
    performance, total_value = calculate_performance(portfolio)
    print_performance(performance, total_value)



def options():
    import yfinance as yf
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    def get_options_chain_for_selected_date(ticker_symbol, selected_index):
        ticker = yf.Ticker(ticker_symbol)

        # Fetch available expiration dates
        expiration_dates = ticker.options

        # Check if the selected index is valid
        if selected_index < 0 or selected_index >= len(expiration_dates):
            print("Invalid selection. Please choose a number from the list.")
            return

        # Get the selected expiration date
        expiration_date = expiration_dates[selected_index]
        print(f"\nFetching options chain for {ticker_symbol} on {expiration_date}...")

        # Fetch the options chain for the selected expiration date
        option_chain = ticker.option_chain(expiration_date)

        # Set pandas display options to show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)  # Ensures wide tables are not truncated

        # Print call options
        print(f"\nCall options for {ticker_symbol} on {expiration_date}:")
        print(option_chain.calls)

        # Print put options
        print(f"\nPut options for {ticker_symbol} on {expiration_date}:")
        print(option_chain.puts)

        # Reset pandas display options to default after use
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')

        # Prompt user to select an option based on the contract symbol
        contract_symbol = input("\nEnter the contract symbol of the option to analyze: ").strip()

        # Check if the contract symbol is valid and select the option data
        if contract_symbol in option_chain.calls['contractSymbol'].values:
            option_data = option_chain.calls[option_chain.calls['contractSymbol'] == contract_symbol]
            option_type = 'Call'
        elif contract_symbol in option_chain.puts['contractSymbol'].values:
            option_data = option_chain.puts[option_chain.puts['contractSymbol'] == contract_symbol]
            option_type = 'Put'
        else:
            print("Invalid contract symbol. Please try again.")
            return

        # Generate the payoff diagram
        strike_price = option_data['strike'].values[0]
        premium = option_data['lastPrice'].values[0]

        # Get the range of stock prices for the plot
        stock_prices = np.linspace(strike_price - 50, strike_price + 50, 100)
        payoff = np.maximum(stock_prices - strike_price, 0) - premium if option_type == 'Call' else np.maximum(strike_price - stock_prices, 0) - premium

        # Plot the payoff diagram
        plt.figure(figsize=(12, 8))
        
        # Plotting call payoff with conditional coloring
        plt.plot(stock_prices, payoff, label=f'{option_type} Option Payoff')

        # Shade the area between the y=0 line and the payoff line
        plt.fill_between(stock_prices, payoff, where=(payoff > 0), color='green', alpha=0.3)
        plt.fill_between(stock_prices, payoff, where=(payoff < 0), color='red', alpha=0.3)

        # Add horizontal line at y=0
        plt.axhline(0, color='orange', linestyle='--', linewidth=1.5, label='Break-Even Point')

        # Add vertical line at the strike price
        plt.axvline(strike_price, color='orange', linestyle='--', linewidth=1.5, label=f'Strike Price (${strike_price})')

        plt.title(f'{option_type} Option Payoff Diagram', color='grey')
        plt.xlabel('Stock Price', color='grey')
        plt.ylabel('Payoff ($)', color='grey')
        plt.legend()
        plt.grid(True)
        
        # Customizing colors
        plt.gca().set_facecolor('black')  # Set background color to black
        plt.gca().tick_params(axis='both', colors='grey')  # Set axis and labels to grey
        plt.gcf().patch.set_facecolor('black')  # Set figure background color to black

        # Custom color based on payoff position
        plt.plot(stock_prices[payoff <= 0], payoff[payoff <= 0], color='red', label='Payoff Below Break-Even')
        plt.plot(stock_prices[payoff > 0], payoff[payoff > 0], color='lime', label='Payoff Above Break-Even')

        plt.show(block=False)

    # Prompt user for ticker symbol
    ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()

    # Create an instance of the Ticker class
    ticker = yf.Ticker(ticker_symbol)

    # Fetch available expiration dates
    expiration_dates = ticker.options

    # List expiration dates with corresponding numbers
    print(f"\nAvailable expiration dates for {ticker_symbol}:")
    for index, date in enumerate(expiration_dates):
        print(f"{date} [{index}]")

    # Prompt user to select an expiration date by number
    try:
        selected_index = int(input("Enter the number corresponding to the expiration date: ").strip())
        get_options_chain_for_selected_date(ticker_symbol, selected_index)
    except ValueError:
        print("Invalid input. Please enter a number.")

def sim():
    import numpy as np
    import matplotlib.pyplot as plt
    import yfinance as yf
    import datetime

    # Prompt for user input
    ticker = input("Enter ticker: ").upper()
    days = int(input("How many days: "))
    simulation_type = int(input("Choose simulation type [ 1=CF | 2=CF(VA) | 3=GBM ]: "))

    # Calculate dates dynamically
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')  # Today's date
    start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')  # Date days ago

    # Fetch the historical equity data
    try:
        historical_data = yf.download(ticker, start=start_date, end=end_date)
        if historical_data.empty:
            raise ValueError("No data found for ticker.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        exit()

    # Get the starting price and calculate volatility
    starting_price = historical_data['Adj Close'].iloc[0]
    returns = historical_data['Adj Close'].pct_change().dropna()
    volatility = returns.std()  # Historical volatility (annualized)
    drift = returns.mean()  # Average daily return (drift)

    # Define the number of simulations
    forloops = 100

    # Function to simulate geometric random walk with volatility adjustment
    def simulate_geometric_random_walk_volatility(start_price, days, volatility):
        prices = [start_price]
        for _ in range(days):
            daily_return = np.random.normal(0, volatility)  # Use normal distribution with historical volatility
            prices.append(prices[-1] * (1 + daily_return))
        return prices

    # Function to simulate geometric random walk with coin-flip strategy
    def simulate_geometric_random_walk_coin_flip(start_price, days):
        prices = [start_price]
        for _ in range(days):
            p = np.random.uniform(0, 1)
            if p > 0.5:
                prices.append(prices[-1] * 1.01)  # Increase by 1% for 'heads'
            else:
                prices.append(prices[-1] * 0.99)  # Decrease by 1% for 'tails'
        return prices

    # Function to simulate geometric Brownian motion (GBM)
    def simulate_geometric_brownian_motion(start_price, drift, volatility, days):
        dt = 1/252  # Time increment (daily)
        prices = [start_price]
        for _ in range(days):
            rand_num = np.random.randn()
            daily_return = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * rand_num
            prices.append(prices[-1] * np.exp(daily_return))  # GBM formula for price movement
        return prices

    # Create a plot
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111)

    # Set plot background color and text color
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    colors = plt.cm.jet(np.linspace(0, 1, forloops))

    # Run the chosen simulation type
    for i in range(forloops):
        if simulation_type == 1:
            simulated_prices = simulate_geometric_random_walk_coin_flip(starting_price, days)
        elif simulation_type == 2:
            simulated_prices = simulate_geometric_random_walk_volatility(starting_price, days, volatility)
        elif simulation_type == 3:
            simulated_prices = simulate_geometric_brownian_motion(starting_price, drift, volatility, days)
        else:
            print("Invalid simulation type selected.")
            exit()
        plt.plot(simulated_prices, color=colors[i])

    # Customize the plot
    plt.title(f"Simulated Stock Price Movement for {ticker} over {days} Days", color='grey')
    plt.xlabel('Day', color='grey')
    plt.ylabel('Price', color='grey')
    plt.grid(True, color='grey', linestyle='--')

    # Show plot
    plt.tight_layout()
    plt.show(block=False)


    
def sc():
    import yfinance as yf
    import mplfinance as mpf
    import pandas as pd

    # Calculate RSI function (14-period)
    def calculate_rsi(data, window=14):
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Dictionary to map user input to valid periods and intervals accepted by yfinance
    timeframes = {
        '1d': ('1d', '1m'),    # 1-day chart with 1-minute interval
        '7d': ('5d', '5m'),    # 5-day chart with 5-minute interval (Yahoo Finance's limit)
        '1m': ('1mo', '1d'),   # 1-month chart with daily interval
        '3m': ('3mo', '1d'),   # 3-month chart with daily interval
        '6m': ('6mo', '1d'),   # 6-month chart with daily interval
        'ytd': ('ytd', '1d'),  # Year-to-date chart with daily interval
        '1y': ('1y', '1d'),    # 1-year chart with daily interval
        '2y': ('2y', '1d'),    # 2-year chart with daily interval
        '5y': ('5y', '1d'),    # 5-year chart with daily interval
        '10y': ('10y', '1d')   # Maximum available data
    }

    def plot_stock_chart(ticker, timeframe):
        try:
            # Get the period and interval for the timeframe
            period, interval = timeframes[timeframe]

            # Download historical stock data with appropriate interval
            stock_data = yf.download(ticker, period=period, interval=interval)

            # Check if data is available
            if stock_data.empty:
                print(f"No data available for {ticker} in the selected timeframe.")
                return

            # Calculate RSI
            stock_data['RSI'] = calculate_rsi(stock_data)

            # Create custom market colors (orange for all)
            mc = mpf.make_marketcolors(up='green', down='red', wick='inherit', edge='inherit', volume='darkblue')

            # Create a style based on 'nightclouds' but with modified market colors
            s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

            # Shade areas for RSI: red when RSI is above 70, green when below 30
            rsi_above_70 = stock_data['RSI'].apply(lambda x: x if x > 70 else None)
            rsi_below_30 = stock_data['RSI'].apply(lambda x: x if x < 30 else None)

            # Additional plot for RSI (on its own panel) with red and green shading
            ap_rsi = [
                mpf.make_addplot(stock_data['RSI'], panel=2, color='grey', ylabel='RSI'),
                mpf.make_addplot(rsi_above_70, panel=2, color='red', alpha=1),  # Red fill for RSI > 70
                mpf.make_addplot(rsi_below_30, panel=2, color='green', alpha=1),  # Green fill for RSI < 30
            ]

            # Plot the stock data using mplfinance with the RSI and volume plots
            fig, axes = mpf.plot(stock_data, type='candle', style=s, title='',  # Set title='' to avoid duplicate title
                                addplot=ap_rsi, volume=True, volume_panel=1, tight_layout=True, returnfig=True)

            # Reduce x-axis font size and set color to orange
            for ax in axes:
                ax.tick_params(axis='x', labelsize=8, colors='orange')  # Set x-axis font size to 8 and color to orange
                ax.yaxis.label.set_color('orange')  # Set y-axis label color to orange
                ax.tick_params(axis='y', colors='orange')  # Set y-axis tick label color to orange

            # Manually set title font size and color to orange
            axes[0].set_title(f"{ticker.upper()} - {timeframe} Chart", fontsize=10, color='orange')

            # Get the latest closing price
            latest_price = stock_data['Close'][-1]

            # Add latest price in the top-left corner of the main panel (axes[0] is the price panel) with orange text
            axes[0].text(0.01, 0.95, f"Latest Price: ${latest_price:.2f}", transform=axes[0].transAxes,
                        fontsize=10, color='orange', verticalalignment='top')

            # Show the plot
            mpf.show(block=False)

        except Exception as e:
            print(f"An error occurred: {e}")

    # Input from user
    ticker = input("Enter the stock ticker symbol: ").upper()
    timeframe = input("Enter timeframe (1d, ytd, 1y, 2y, 5y, 10y): ").lower()

    # Validate user input for timeframe
    if timeframe not in timeframes:
        print(f"Invalid timeframe '{timeframe}'. Please enter one of: {', '.join(timeframes.keys())}")
    else:
        plot_stock_chart(ticker, timeframe)


    
def cc():
    import yfinance as yf

    # ANSI color codes
    LIME_GREEN = '\033[92m'
    NEON_RED = '\033[91m'
    BLUE = '\033[94m'  # Blue color for commodity names
    RESET = '\033[0m'

    # Tickers for Commodities and Cryptocurrencies
    commodity_tickers = {
        "S&P 500": "VOO",
        "NASDAQ 100": "QQQ",
        "TQQQ": "TQQQ",
        "QLD": "QLD",
        "NTSX": "NTSX",
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD", 
        "10-Year Treasury Yield": "^TNX", 
        "30-Year Treasury Yield": "^TYX", 
        "Gold": "GC=F"   
    }

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def print_performance(performance):
        for name, data in performance.items():
            print(f"{BLUE}{name} ({data['ticker']}){RESET}")
            print(f"Current Price: ${data['current_price']:.2f}" if data['current_price'] != 'N/A' else "Current Price: N/A")
            print(f"1D %Δ: {colorize_percent(data['percent_change_today'])}" if data['percent_change_today'] != 'N/A' else "Percent Change Today: N/A")
            print()

    def get_data(tickers):
        performance = {}
        for name, ticker in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="1d")
                if not data.empty:
                    current_price = data['Close'].iloc[0]  # Use iloc for positional indexing
                    percent_change_today = ((data['Close'].iloc[0] - data['Open'].iloc[0]) / data['Open'].iloc[0]) * 100
                else:
                    current_price = 'N/A'
                    percent_change_today = 'N/A'
                
                performance[name] = {
                    'ticker': ticker,
                    'current_price': current_price,
                    'percent_change_today': percent_change_today
                }
            except Exception as e:
                performance[name] = {
                    'ticker': ticker,
                    'current_price': 'N/A',
                    'percent_change_today': 'N/A'
                }
                print(f"Error retrieving data for {name}: {e}")
        
        return performance

    # Get and print commodity data
    commodity_performance = get_data(commodity_tickers)
    print_performance(commodity_performance)


def des():
    import yfinance as yf
    from colorama import Fore, Style, init
    import textwrap
    import pandas as pd

    # Initialize colorama
    init(autoreset=True)

    # Define color
    blue = Fore.BLUE

    def print_colored_percentage(change, label):
        if change >= 0:
            color = Fore.GREEN
        else:
            color = Fore.RED
        print(f"{blue}{label}: {color}{change:.2f}%{Style.RESET_ALL}")

    def get_stock_info(ticker):
        stock = yf.Ticker(ticker)
        
        # Get stock info
        info = stock.info
        price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        
        # Calculate percent changes
        history = stock.history(period="5y")
        if len(history) == 0:
            print("No historical data available.")
            return
        
        today_price = history.iloc[-1]['Close']

        # Safe calculation with available data
        def safe_calculate_change(past_price, current_price):
            return ((current_price - past_price) / past_price) * 100 if past_price else 0

        # 1 Day Change
        one_day_change = safe_calculate_change(history.iloc[-2]['Close'] if len(history) > 1 else today_price, today_price)
        
        # 6 Months Change
        six_months_ago_index = -126 if len(history) > 126 else 0
        six_months_ago_price = history.iloc[six_months_ago_index]['Close']
        six_month_change = safe_calculate_change(six_months_ago_price, today_price)

        # YTD Change
        start_of_year_price = history[history.index.year == history.index[-1].year].iloc[0]['Close'] if len(history) > 0 else today_price
        ytd_change = safe_calculate_change(start_of_year_price, today_price)
        
        # 3 Years Change
        three_years_ago_index = history.index <= (history.index[-1] - pd.DateOffset(years=3))
        if any(three_years_ago_index):
            three_years_ago_price = history.loc[three_years_ago_index].iloc[0]['Close']
            three_year_change = safe_calculate_change(three_years_ago_price, today_price)
        else:
            three_year_change = 0

        # Print stock info
        description = info.get('longBusinessSummary', 'Description not available.')
        print(f"\n{blue}Description:{Style.RESET_ALL}\n{textwrap.fill(description, width=80)}")
        print(f"{blue}Price:{Style.RESET_ALL} ${price:.2f}")
        print(f"{blue}Market Cap:{Style.RESET_ALL} ${market_cap / 1e9:.2f} Billion")

        # Print percent changes
        print_colored_percentage(one_day_change, "1D Change")
        print_colored_percentage(six_month_change, "6M Change")
        print_colored_percentage(ytd_change, "YTD Change")
        print_colored_percentage(three_year_change, "3Y Change")

    if __name__ == "__main__":
        ticker = input("Enter stock ticker: ").upper()
        get_stock_info(ticker)

def ovs():
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import datetime as dt
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap

    def option_chains(ticker):
        """
        Fetches option chains for all expiration dates for a given ticker symbol.
        """
        asset = yf.Ticker(ticker)
        expirations = asset.options
        chains = pd.DataFrame()

        for expiration in expirations:
            opt = asset.option_chain(expiration)
            calls = opt.calls
            calls['optionType'] = "call"
            puts = opt.puts
            puts['optionType'] = "put"
            chain = pd.concat([calls, puts])
            chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
            chains = pd.concat([chains, chain])

        chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
        return chains

    def plot_volatility_surface(ax, options, min_date, max_date, option_type='call'):
        """
        Plots a 3D surface of implied volatility based on the selected date range and option type (call or put).
        """
        # Filter data based on the selected range and option type
        expiration_dates = pd.to_datetime(options['expiration'])
        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)
        
        mask = (expiration_dates >= min_date) & (expiration_dates <= max_date) & (options['optionType'] == option_type)
        filtered_options = options[mask]

        # Pivot the dataframe to prepare for 3D plotting
        surface = (
            filtered_options[['daysToExpiration', 'strike', 'impliedVolatility']]
            .pivot_table(values='impliedVolatility', index='strike', columns='daysToExpiration')
            .dropna()
        )

        # Get the 1D values from the pivoted dataframe
        x, y, z = surface.columns.values, surface.index.values, surface.values

        # Return coordinate matrices from coordinate vectors
        X, Y = np.meshgrid(x, y)

        # Set labels
        ax.set_xlabel('Days to Expiration', color='grey')
        ax.set_ylabel('Strike Price', color='grey')
        ax.set_zlabel('Implied Volatility', color='grey')
        ax.set_title(f'{option_type.capitalize()} Implied Volatility Surface', color='grey')

        # Define custom color map with the requested gradient
        colors = ["#00008B", "#00FFFF", "#008080", "#00FF00", "#FFFF00", "#FFA500", "#FF0000", "#8B0000"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        # Plot
        surf = ax.plot_surface(X, Y, z, cmap=custom_cmap, edgecolor='none')

        # Customize appearance
        ax.set_facecolor('black')  # Set background color of the axes
        ax.tick_params(axis='both', colors='grey')  # Set tick colors
        ax.xaxis.pane.fill = False  # Hide the pane
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, color='grey')  # Set grid color

        return surf

    def main():
        # Prompt user for ticker symbol
        ticker_symbol = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()

        # Get option chain data
        options = option_chains(ticker_symbol)

        # List available expiration dates
        expiration_dates = pd.to_datetime(options['expiration']).sort_values().unique()
        print("\nAvailable expiration dates:")
        for index, date in enumerate(expiration_dates):
            print(f"{index}: {date.date()}")

        # Prompt user to select a range of expiration dates
        try:
            min_index = int(input("Enter the index of the earliest expiration date: ").strip())
            max_index = int(input("Enter the index of the latest expiration date: ").strip())

            if min_index < 0 or max_index >= len(expiration_dates) or min_index > max_index:
                print("Invalid range selected.")
                return

            min_date = expiration_dates[min_index]
            max_date = expiration_dates[max_index]

            # Filter options for plotting
            calls = options[options["optionType"] == "call"]
            puts = options[options["optionType"] == "put"]

            # Create a figure with two 3D subplots
            fig = plt.figure(figsize=(18, 8))
            fig.patch.set_facecolor('black')  # Set background color of the figure

            # Call volatility surface
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = plot_volatility_surface(ax1, calls, min_date, max_date, 'call')

            # Put volatility surface
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = plot_volatility_surface(ax2, puts, min_date, max_date, 'put')

            # Add color bars
            fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
            fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

            plt.show(block=False)
        
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

    if __name__ == "__main__":
        main()

def val():
    import webbrowser
    def parse_number_input(input_str):
        # Handle M for million and B for billion
        if input_str[-1].upper() == 'M':
            return float(input_str[:-1]) * 1_000_000
        elif input_str[-1].upper() == 'B':
            return float(input_str[:-1]) * 1_000_000_000
        else:
            return float(input_str)

    ticker = input("Enter a stock ticker: ").strip()

    # Automatically open the relevant links with the ticker included
    roic_link = f"https://www.roic.ai/quote/{ticker}/financials"
    stockanalysis_link = f"https://stockanalysis.com/stocks/{ticker}/financials/"

    webbrowser.open(roic_link)
    webbrowser.open(stockanalysis_link)

    # Allow the user to enter values manually
    latest_price = parse_number_input(input("share $: ").strip())
    market_cap = parse_number_input(input("MC: ").strip())
    #ttm_gross_profit = parse_number_input(input("TTM Gross Profit: ").strip())
    ttm_free_cash_flow = parse_number_input(input("TTM FCF: ").strip())
    shares_outstanding = parse_number_input(input("Total Share #: ").strip())
    #book_value = parse_number_input(input("BV/SH: ").strip())

    # Calculate FCF per Share
    fcf_per_share = round(ttm_free_cash_flow / shares_outstanding, 2)
    fcf_per_share_display = f"${fcf_per_share:,.2f}"

    # Calculate Price/FCF (Market Cap / TTM Free Cash Flow)
    price_fcf = round(market_cap / ttm_free_cash_flow, 2)
    price_fcf_display = f"{price_fcf:,.2f}"

    # Print all information
    print(f"\nTicker: {ticker.upper()}")
    print(f"Price: ${latest_price:,.2f}")
    print(f"MC: ${market_cap:,.0f}")
    #print(f"TTM Profit: ${ttm_gross_profit:,.0f}")
    print(f"TTM Free Cash Flow: ${ttm_free_cash_flow:,.0f}")
    print(f"Shares Issued: {shares_outstanding:,.0f}")
    #print(f"BV/SH: ${book_value:,.2f}")
    print(f"FCF/SH: {fcf_per_share_display}")
    print(f"Price/FCF: {price_fcf_display}")

    # Prompt for multiple growth rates
    growth_rates_input = input("Enter growth rates to test (%) : ").strip()
    growth_rates = [float(rate.strip()) for rate in growth_rates_input.split(',')]

    # Calculate and print future FCF and final value for each growth rate
    for growth_rate in growth_rates:
        r = (growth_rate / 100)
        n = 10  # 10 years
        future_fcf = ttm_free_cash_flow * (1 + r) ** n
        final_value = (future_fcf * price_fcf) / shares_outstanding
        percent_return = ((final_value - latest_price) / latest_price) * 100
        cagr = (percent_return / 100 + 1) ** (1 / n) - 1
        cagr = cagr * 100

        # Display calculations
        print(f"\nGrowth Rate: {growth_rate:.2f}%")
        print(f"Predicted FCF in 10 Years: ${round(future_fcf):,}")
        print(f"Final Value ((Future FCF * Price/FCF) / Shares Issued): ${round(final_value):,}")
        print(f"Percent Return: {percent_return:.2f}%")
        print(f"CAGR: {cagr:.2f}%\n")

   # if __name__ == "__main__":
        #val()


def dcf():
    import webbrowser
    import sys
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq

    def parse_number_input(input_str):
        try:
            input_str = input_str.upper()
            if input_str.endswith('M'):
                return float(input_str[:-1]) * 1_000_000
            elif input_str.endswith('B'):
                return float(input_str[:-1]) * 1_000_000_000
            else:
                return float(input_str)
        except ValueError:
            print("Invalid number format. Please enter a valid number.")
            sys.exit()

    def compute_intrinsic_value(ttm_fcf, fcf_growth_rate, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding):
        projected_fcfs = []
        discounted_fcfs = []
        for i in range(projection_years):
            fcf = ttm_fcf * (1 + fcf_growth_rate) ** (i + 1)
            projected_fcfs.append(fcf)
            discounted_fcf = fcf / (1 + wacc) ** (i + 1)
            discounted_fcfs.append(discounted_fcf)

        terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth_rate)
        if wacc <= terminal_growth_rate:
            print("WACC must be greater than the terminal growth rate.")
            return None
        terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
        discounted_terminal_value = terminal_value / (1 + wacc) ** projection_years

        enterprise_value = sum(discounted_fcfs) + discounted_terminal_value
        equity_value = enterprise_value - net_debt

        intrinsic_value_per_share = equity_value / shares_outstanding
        return intrinsic_value_per_share

    def calculate_average_growth_rates(growth_rates):
        if len(growth_rates) < 2:
            return None, None
        
        two_year_growth = np.mean(growth_rates[:2])
        six_year_growth = np.mean(growth_rates[:6]) if len(growth_rates) >= 6 else None
        
        return two_year_growth, six_year_growth

    def plot_intrinsic_value(growth_rates, intrinsic_values, ticker, latest_price):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot settings
        ax.plot(growth_rates, intrinsic_values, color='orange', linewidth=2, label='Intrinsic Value')
        ax.axhline(y=latest_price, color='red', linestyle='--', linewidth=2, label='Current Price')
        
        # Customize colors
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Customize grid
        ax.grid(True, color='orange', linestyle=':', alpha=0.3)
        
        # Customize axes
        ax.spines['bottom'].set_color('orange')
        ax.spines['top'].set_color('orange')
        ax.spines['left'].set_color('orange')
        ax.spines['right'].set_color('orange')
        ax.tick_params(axis='x', colors='orange')
        ax.tick_params(axis='y', colors='orange')
        
        # Labels and title
        ax.set_xlabel('Growth Rate (%)', color='orange', fontsize=12)
        ax.set_ylabel('Intrinsic Value per Share ($)', color='orange', fontsize=12)
        ax.set_title(f'Intrinsic Value vs Growth Rate for {ticker.upper()}', color='orange', fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(facecolor='black', edgecolor='orange', labelcolor='orange')
        
        plt.tight_layout()
        plt.show()

    def calculate_required_growth_rate(ttm_fcf, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding, current_price):
        def intrinsic_value_difference(growth_rate):
            intrinsic_value = compute_intrinsic_value(ttm_fcf, growth_rate, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding)
            return intrinsic_value - current_price

        try:
            required_growth_rate = brentq(intrinsic_value_difference, -0.99, 1.0)
            return required_growth_rate
        except ValueError:
            return None

    def calculate_intrinsic_value():
        # Prompt for ticker symbol and open financial links
        ticker = input("Enter a stock ticker: ").strip()
        roic_link = f"https://www.roic.ai/quote/{ticker}/financials"
        stockanalysis_link = f"https://stockanalysis.com/stocks/{ticker}/financials/"
        webbrowser.open(roic_link)
        webbrowser.open(stockanalysis_link)

        # Get historical growth rates
        print("Enter historical FCF growth rates (most recent first, separated by tabs or commas):")
        historical_rates_input = input().strip()
        # Split on tabs and commas
        historical_rates = re.split(r'[,\t]+', historical_rates_input)
        historical_rates = [float(rate.strip().rstrip('%')) / 100 for rate in historical_rates]

        # Calculate average growth rates
        two_year_growth, six_year_growth = calculate_average_growth_rates(historical_rates)
        
        if two_year_growth is not None:
            print(f"2-year average FCF growth rate: {two_year_growth*100:.2f}%")
        if six_year_growth is not None:
            print(f"6-year average FCF growth rate: {six_year_growth*100:.2f}%")

        # Gather required inputs
        latest_price = parse_number_input(input("Enter the latest stock price: ").strip())
        ttm_free_cash_flow = parse_number_input(input("Enter the TTM Free Cash Flow (e.g., 300M or 0.3B): ").strip())
        shares_outstanding = parse_number_input(input("Enter the number of shares outstanding (e.g., 1000M or 1B): ").strip())
        net_debt = parse_number_input(input("Enter the Net Debt (e.g., 100M or 0.1B): ").strip())
        wacc = float(input("Enter the Discount Rate (WACC) (e.g., 8 for 8%): ").strip()) / 100
        terminal_growth_rate = float(input("Enter the Terminal Growth Rate (e.g., 3 for 3%): ").strip()) / 100
        projection_years = int(input("Enter the number of projection years (e.g., 10): ").strip())

        # Calculate required growth rate
        required_growth_rate = calculate_required_growth_rate(ttm_free_cash_flow, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding, latest_price)
        if required_growth_rate is not None:
            print(f"\nRequired FCF growth rate for intrinsic value to equal current price: {required_growth_rate*100:.2f}%")
        else:
            print("\nUnable to calculate the required growth rate. The current price might be outside the calculable range.")

        # Get growth rate range for plotting
        growth_range = input("Enter the growth rate range for plotting (e.g., -5,50 for -5% to 50%): ").strip()
        min_growth, max_growth = map(float, growth_range.split(','))
        growth_rates_plot = np.linspace(min_growth/100, max_growth/100, 100)

        # Get multiple growth rates for specific calculations
        growth_rates = []
        print("Enter FCF growth rates to test (e.g., 5 for 5%). Enter multiple rates separated by commas, or press Enter to finish:")
        rates_input = input().strip()
        if rates_input:
            growth_rates = [float(rate.strip()) / 100 for rate in rates_input.split(',')]

        # Add historical average growth rates if available
        if two_year_growth is not None:
            growth_rates.append(two_year_growth)
        if six_year_growth is not None:
            growth_rates.append(six_year_growth)

        # Calculate and display results
        print("\nResults:")
        print("Growth Rate | Intrinsic Value per Share")
        print("-" * 40)

        for rate in growth_rates:
            intrinsic_value = compute_intrinsic_value(ttm_free_cash_flow, rate, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding)
            if intrinsic_value is not None:
                print(f"{rate*100:9.2f}% | ${intrinsic_value:,.2f}")

        print(f"\nCurrent Stock Price: ${latest_price:,.2f}")

        # Calculate intrinsic values for plotting
        intrinsic_values_plot = [compute_intrinsic_value(ttm_free_cash_flow, rate, wacc, terminal_growth_rate, projection_years, net_debt, shares_outstanding) for rate in growth_rates_plot]

        # Plot the results
        plot_intrinsic_value(growth_rates_plot * 100, intrinsic_values_plot, ticker, latest_price)

    if __name__ == "__main__":
        calculate_intrinsic_value()

def fs():
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    from io import StringIO
    from colorama import Fore, Style

    # Set pandas options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    def fetch_financials(ticker, url_suffix):
        url = f"https://stockanalysis.com/stocks/{ticker}/financials/{url_suffix}"
        response = requests.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = soup.find_all('table')
            
            if tables:
                table_html = StringIO(str(tables[0]))
                df = pd.read_html(table_html)[0]
                
                # Remove the last column
                df = df.iloc[:, :-1]
                
                return df
            else:
                print(f"No tables found on the page: {url}")
                return None
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None

    def print_separator():
        print('-' * 80)

    def format_value(value):
        if isinstance(value, str) and value.startswith('-') and '%' in value:
            return f"{Fore.RED}{value}{Style.RESET_ALL}"
        else:
            return value

    def print_df_with_colored_first_column(df):
        if df is not None and not df.empty:
            # Determine the width of the first column based on the longest item
            max_length = df.iloc[:, 0].astype(str).map(len).max() + 2
            
            # Calculate column widths for the rest of the columns
            col_widths = [len(str(item)) for item in df.iloc[:, 1:].values.flatten()]
            max_col_width = max(col_widths) + 2
            
            for index, row in df.iterrows():
                # Format columns with fixed width first
                first_col = f"{row.iloc[0]:<{max_length}}"
                rest_cols = [f"{str(item):<{max_col_width}}" for item in row.iloc[1:]]
                
                # Apply color formatting
                first_col_colored = f"{Fore.BLUE}{first_col}{Style.RESET_ALL}"
                rest_cols_colored = [format_value(col) for col in rest_cols]
                
                print(f"{first_col_colored} " + " ".join(rest_cols_colored))
        else:
            print("No data to display.")

    # Prompt the user to enter the ticker symbol
    ticker = input("Enter the stock ticker symbol: ").strip().upper()

    # List of URL suffixes and their descriptions
    url_suffixes = [
        ("", "Income Statement"),
        ("balance-sheet/", "Balance Sheet"),
        ("cash-flow-statement/", "Cash Flow Statement"),
        ("ratios/", "Financial Ratios")
    ]

    # Fetch and print financial data for each URL
    for suffix, description in url_suffixes:
        print(f"\n{description}:")
        print_separator()
        financial_data = fetch_financials(ticker, suffix)
        print_df_with_colored_first_column(financial_data)
        print_separator()

def cl():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import webbrowser
    from tabulate import tabulate

    # ANSI escape codes for colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    # Function to format cells with color based on the number
    def color_numbers(value):
        try:
            # Remove non-numeric characters (e.g., $ and ,) and convert to float
            number = float(value.replace('$', '').replace(',', ''))
            # Check if the number is positive or negative
            if number > 0:
                return GREEN + value + RESET
            elif number < 0:
                return RED + value + RESET
        except ValueError:
            # Return the value as is if it's not a number
            return value
        return value

    # Step 1: Send an HTTP request to the website
    url = "http://openinsider.com/"
    response = requests.get(url)

    # Step 2: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 3: Find all tables on the page
    tables = soup.find_all('table', class_='tinytable')  # Adjust class or identifier if necessary

    # Force pandas to show all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')

    # Step 4: Iterate through all the tables
    for idx, table in enumerate(tables):
        # Skip Table 4 (index is 3 since index starts from 0)
        if idx == 3:
            continue

        print(f"Table {idx + 1}:")

        # Extract the table headers
        headers = [th.text.strip() for th in table.find_all('th')]

        # Extract the rows and corresponding data
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip the header row
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:  # Only append non-empty rows
                rows.append(cells)

        # Step 5: Convert the data into a pandas DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Step 6: Drop the specified columns if they exist in the DataFrame
        columns_to_drop = ['1d', '1w', '1m', '6m']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Step 7: Apply color formatting to positive and negative numbers
        df = df.apply(lambda col: col.map(color_numbers))

        # Step 8: Output the entire DataFrame with reduced cell padding
        print(tabulate(df, headers='keys', tablefmt='plain', showindex=False))

        print("\n" + "="*80 + "\n")  # Separator between tables

    # Step 9: Prompt to open the website link in the browser
    open_link = input("Would you like to open the website in your browser? (y/n): ").strip().lower()
    if open_link == 'y':
        webbrowser.open(url)

def fund():
    import requests
    from bs4 import BeautifulSoup
    import webbrowser

    def scrape_holdings(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all tables on the page
        tables = soup.find_all('table')
        print(f"Found {len(tables)} table(s) on the page.")
        
        if tables:
            table = tables[0]
            
            # Extract table rows
            rows = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cols = [col.text.strip() for col in row.find_all('td')]
                if cols:  # Only add rows that contain columns
                    rows.append(cols)
            
            # Check if rows are empty
            if not rows:
                print("No data rows found in the table.")
                return
            
            # Determine maximum column widths
            num_cols = max(len(row) for row in rows)
            max_col_widths = [0] * num_cols
            
            for row in rows:
                for i, col in enumerate(row):
                    max_col_widths[i] = max(max_col_widths[i], len(col))
            
            # Print the results with aligned columns
            print("\nTable Data:")
            for row in rows:
                # Ensure all rows have the same number of columns
                row = row + [''] * (num_cols - len(row))
                print(" | ".join(f"{col:<{max_col_widths[i]}}" for i, col in enumerate(row)))
        else:
            print("No tables found on the page.")
        
        # Ask user if they want to open the URL
        open_link = input(f"Do you want to open the URL in your browser? (y/n): ").strip().lower()
        if open_link == 'y':
            webbrowser.open(url)
        else:
            print("The URL was not opened.")

    def main():
        fund = input("Enter the fund name: ").strip().lower()
        if fund == "scion":
            url = "https://www.dataroma.com/m/holdings.php?m=SAM"
        elif fund == "icahn":
            url = "https://www.dataroma.com/m/holdings.php?m=ic"
        elif fund == "ackman":
            url = "https://www.dataroma.com/m/holdings.php?m=psc"
        elif fund == "brk":
            url = "https://www.dataroma.com/m/holdings.php?m=BRK"
        elif fund == "big bets":
            url = "https://www.dataroma.com/m/g/portfolio.php?o=b"
        elif fund == "low":
            url = "https://www.dataroma.com/m/g/portfolio.php?pct=5&o=ru"
        else:
            print("No action defined for this fund.")
            return
        
        scrape_holdings(url)

    if __name__ == "__main__":
        main()

def portchart():
    import json
    import yfinance as yf
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num
    import matplotlib.dates as mdates

    # Load portfolio data
    with open('portfolio.json', 'r') as file:
        portfolio_data = json.load(file)

    # Extract tickers, skipping '1'
    tickers = [ticker for ticker in portfolio_data.keys() if ticker != '1']

    # Define chart parameters
    num_stocks = len(tickers)
    grid_size = int(num_stocks**0.5) + 1
    fig_width = 10  # Adjust width to fit more charts
    fig_height = 10  # Adjust height to fit more charts
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_width, fig_height), facecolor='black')
    axes = axes.flatten()  # Flatten to make indexing easier

    # Custom style settings
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': 'black',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'orange',
        'xtick.color': 'orange',
        'ytick.color': 'orange',
        'grid.color': 'orange',
        'figure.facecolor': 'black',
        'figure.edgecolor': 'black',
        'axes.grid': True,
        'axes.labelsize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6
    })

    for i, ticker in enumerate(tickers):
        prices = yf.Ticker(ticker).history(period='ytd')

        # Plot line chart
        ax = axes[i]
        ax.plot(prices.index, prices['Close'], color='orange', label=ticker)
        ax.fill_between(prices.index, prices['Close'], color='darkblue', alpha=0.5)
        ax.set_title(ticker, color='orange', fontsize=8)
        ax.set_xlabel('Date', color='orange', fontsize=8)
        ax.set_ylabel('Close Price', color='orange', fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(True, linestyle='--')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=45, color='orange')
        plt.setp(ax.get_yticklabels(), color='orange')
        ax.patch.set_facecolor('black')  # Set the axes background color

    # Hide unused subplots
    for j in range(num_stocks, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)  # Adjust padding to fit the plots
    plt.show(block=False)

def sec():
    import webbrowser

    def open_sec_filing(ticker, filing_type):
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        if filing_type == 'k':
            url = f"{base_url}?action=getcompany&CIK={ticker}&type=10-k&dateb=&owner=exclude&count=40"
        elif filing_type == 'q':
            url = f"{base_url}?action=getcompany&CIK={ticker}&type=10-q&dateb=&owner=exclude&count=40"
        else:
            print("Invalid filing type. Please enter 'k' for 10-K or 'q' for 10-Q.")
            return
        webbrowser.open(url)
        print(f"Opening {ticker} {filing_type}")
    def main():
        ticker = input("Enter ticker symbol: ").strip().upper()
        filing_type = input("Enter 'k' for 10-K or 'q' for 10-Q: ").strip().lower()
        open_sec_filing(ticker, filing_type)
    if __name__ == "__main__":
        main()

def seclist():
    from datetime import datetime
    import logging
    import webbrowser  # To open links in the browser

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # ANSI escape codes for text colors
    RED = '\033[91m'
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;208m'  # Orange (approximation)
    RESET = '\033[0m'

    def get_company_filings(ticker):
        try:
            # Set the identity with the provided email
            set_identity("rishisraja0@gmail.com")

            # Get the company
            company = Company(ticker)

            # Get 10-K and 10-Q filings
            filings = company.get_filings(form=["10-K", "10-Q"])

            # Prepare the results list
            results = []

            # Iterate through the filings and extract required information
            for filing in filings:
                try:
                    # Get the filing object
                    obj = filing.obj()
                    
                    # Extract the period of report
                    if hasattr(obj, 'period_of_report'):
                        period_of_report = obj.period_of_report
                    elif hasattr(filing, 'filing_date'):
                        period_of_report = filing.filing_date
                    else:
                        period_of_report = "Not available"
                    
                    # Format the date if it's a datetime object
                    if isinstance(period_of_report, datetime):
                        period_of_report = period_of_report.strftime('%Y-%m-%d')

                    # Construct the correct filing link
                    filing_link = f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={filing.cik}&accession_number={filing.accession_number}&xbrl_type=v"
                    
                    # Construct the EDGAR browse link for the CIK
                    cik_browse_link = f"https://www.sec.gov/edgar/browse/?CIK={filing.cik}&owner=exclude"

                    # Append the filing information to the results list
                    results.append({
                        'form': filing.form,
                        'cik': filing.cik,
                        'accession_number': filing.accession_number,
                        'period_of_report': period_of_report,
                        'filing_link': filing_link,
                        'cik_browse_link': cik_browse_link
                    })
                except AttributeError as e:
                    logging.warning(f"Skipping a filing due to missing attribute: {str(e)}")
                    continue

            return results

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None

    def main():
        ticker = input("Enter a stock ticker: ").upper()
        filings = get_company_filings(ticker)

        if filings:
            print(f"\nFilings for {ticker}:")
            for i, filing in enumerate(filings, 1):
                # Set the color based on the form type
                form_color = BLUE if filing['form'] == '10-Q' else ORANGE if filing['form'] == '10-K' else RESET
                # Print the filing information with colors
                print(f"{RED}[{i}]{RESET}Form: {form_color}{filing['form']}{RESET}")
                print(f"CIK: {filing['cik']}")
                print(f"Accession Number: {filing['accession_number']}")
                print(f"Filing Date: {filing['period_of_report']}")
                #print(f"Filing Link: {filing['filing_link']}")
                #print(f"CIK Browse Link: {filing['cik_browse_link']}")
                print()  # Add an empty line between filings for better readability

            # Prompt the user to select a filing
            choice = input("Enter the number of the filing you want to open (or press Enter to skip): ")
            if choice.isdigit() and 1 <= int(choice) <= len(filings):
                selected_filing = filings[int(choice) - 1]
                print(f"Opening {selected_filing['form']} for period {selected_filing['period_of_report']}")

                # Open the filing link
                webbrowser.open(selected_filing['filing_link'])

                # Open the CIK browse link
                webbrowser.open(selected_filing['cik_browse_link'])
            else:
                print("No valid filing selected or skipping.")
        else:
            print(f"No filings found for {ticker}")

    if __name__ == "__main__":
        main()

def main():
    while True:
        print("\nMenu:")
        print("[ch][news][cc][gm][des][wl]")
        print("[sa][fv][hol][ins][roic][fs]")
        print("[port][10k][10q][op][sim][ovs]")
        print("[vic][gain][val][dcf][sc][cl]")
        print("[fund][pch][sec][secl][pn][screen]")

        choice = input("Choose an option: ").strip()
        
        if choice == 'ch':
            plot()
        elif choice == 'news':
            import webbrowser
            webbrowser.open("https://finviz.com/news.ashx")
        elif choice == 'cc':
            cc()
        elif choice == 'gm':
            gm()
        elif choice == 'sa':
            ticker = input("Enter ticker (or press Enter to open homepage): ").strip().upper()
            if ticker:
                url = f"https://seekingalpha.com/symbol/{ticker}"
            else:
                url = "https://seekingalpha.com/"
            # Open the URL in the default web browser
            import webbrowser
            webbrowser.open(url)
        elif choice == 'fv':
            ticker = input("Enter ticker (or press Enter to open homepage): ").strip().upper()
            if ticker:
                url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
            else:
                url = "https://finviz.com/"
            # Open the URL in the default web browser
            import webbrowser
            webbrowser.open(url)
        elif choice == '10k':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-K"
            import webbrowser
            webbrowser.open(url)
        elif choice == '10q':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-Q"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'hol':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://whalewisdom.com/stock/{ticker}"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'ins':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"http://openinsider.com/search?q={ticker}"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'fs':
            fs()
        elif choice == 'roic':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'port':
            port()
        elif choice == 'des':
            des()
        elif choice == 'wl':
            import webbrowser
            webbrowser.open("https://finviz.com/portfolio.ashx?pid=1909245")
        elif choice == 'op':
            options()
        elif choice == 'sim':
            sim()
        elif choice == 'ovs':
            ovs()
        elif choice == 'vic':
            import webbrowser
            webbrowser.open("https://valueinvestorsclub.com/ideas")
        elif choice == 'gain':
            import webbrowser
            webbrowser.open("https://stockanalysis.com/markets/gainers/month/")
        elif choice == 'val':
            val()
        elif choice == 'dcf':
            dcf()
        elif choice == 'sc':
            sc()
        elif choice == 'cl':
            cl()
        elif choice == 'fund':
            fund()
        elif choice == 'pch':
            portchart()
        elif choice == 'sec':
            sec()
        elif choice == 'secl':
            seclist()
        elif choice == 'screen':
            import webbrowser
            webbrowser.open("https://finviz.com/screener.ashx")
        elif choice == 'pn':
            import webbrowser
            webbrowser.open("https://finviz.com/portfolio.ashx?v=1&pid=1911250")
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
