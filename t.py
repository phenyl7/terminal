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
    RED = "\033[91m"
    BLUE = "\033[38;5;24m"
    ORANGE = "\033[38;5;214m"
    GREEN = "\033[38;5;22m"
    LIGHT_GRAY = "\033[38;5;250m"
    DARK_GRAY = "\033[38;5;235m"
    BROWN = "\033[38;5;130m"
    RESET = "\033[0m"

    import matplotlib.pyplot as plt
    import yfinance as yf
    from datetime import datetime, timedelta

    def plot_stock_price():
        # Specify ticker symbol
        ticker = input("Enter the stock ticker: ")

        # Prompt user for time range
        print(f"{BLUE}Choose period:{RESET}")
        print(f"{RED}[4]{RESET} {BROWN}6m{RESET}")
        print(f"{RED}[5]{RESET} {BROWN}YTD{RESET}")
        print(f"{RED}[6]{RESET} {BROWN}1Y{RESET}")
        print(f"{RED}[7]{RESET} {BROWN}3Y{RESET}")
        print(f"{RED}[8]{RESET} {BROWN}5Y{RESET}")
        print(f"{RED}[9]{RESET} {BROWN}10Y{RESET}")
        
        time_range_number = input("choose: ")

        # Set the date range and interval based on user input
        end_date = datetime.now()
        if time_range_number == '4':
            start_date = end_date - timedelta(days=180)
            interval = '1d'
        elif time_range_number == '5':
            start_date = datetime(end_date.year, 1, 1)
            interval = '1d'
        elif time_range_number == '6':
            start_date = end_date - timedelta(days=365)
            interval = '1d'
        elif time_range_number == '7':
            start_date = end_date - timedelta(days=3*365)
            interval = '1d'
        elif time_range_number == '8':
            start_date = end_date - timedelta(days=5*365)
            interval = '1d'
        elif time_range_number == '9':
            start_date = end_date - timedelta(days=10*365)
            interval = '1d'
        else:
            print("Defaulting to 5 year.")
            start_date = end_date - timedelta(days=5*365)
            interval = '1d'

        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

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

        ax1[0].set_title(f'{ticker} {time_range_number}', color='white')
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
    from tabulate import tabulate

    # Define ANSI color codes
    ORANGE = "\033[38;5;130m"
    LIME_GREEN = '\033[1;32m'
    NEON_RED = '\033[91m'
    WHITE = '\033[97m'
    GRAY = "\033[38;5;250m"
    DARK_GRAY = "\033[30m"  # Added dark gray color for table gridlines
    BLUE = "\033[38;5;24m"  # Header color
    RESET = '\033[30m'

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
        
        # Latest price and today's data
        latest_price = data['Close'].iloc[-1] if not data.empty else None
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

    def color_structural_gridlines(table_string, color):
        lines = table_string.split('\n')
        colored_lines = []
        for i, line in enumerate(lines):
            if i == 0 or i == len(lines) - 1 or set(line) <= set('+-=|'):  # First, last, or separator lines
                colored_line = color + ''.join([char if char in '+-=|' else RESET + char + color for char in line]) + RESET
            else:
                colored_line = color + '|' + RESET + line[1:-1] + color + '|' + RESET
            colored_lines.append(colored_line)
        return '\n'.join(colored_lines)

    def print_performance(performance, total_value):
        headers = [f"{GRAY}Ticker{RESET}", f"{GRAY}Shares{RESET}", f"{GRAY}Cost Basis{RESET}", f"{GRAY}Current Price{RESET}", f"{GRAY}Δ{RESET}", f"{GRAY}1D %Δ{RESET}", f"{GRAY}Value{RESET}"]
        table_data = []

        for ticker, data in performance.items():
            row = [
                f"{ORANGE}{ticker.upper()}{RESET}",
                f"{ORANGE}{data['shares']}{RESET}",
                f"{LIME_GREEN}${data['cost_basis']:.2f}{RESET}",
                f"{LIME_GREEN}${data['current_price']:.2f}" if data['current_price'] != 'N/A' else "N/A",
                colorize_percent(data['percent_change']) if data['percent_change'] != 'N/A' else "N/A",
                colorize_percent(data['percent_change_today']) if data['percent_change_today'] != 'N/A' else "N/A",
                f"{LIME_GREEN}${data['value']:.2f}" if data['value'] != 'N/A' else "N/A"
            ]
            table_data.append(row)

        table = tabulate(table_data, headers=headers, tablefmt="grid")
        colored_table = color_structural_gridlines(table, DARK_GRAY)
        print(colored_table)
        print(f"\n {GRAY}Total Portfolio Value: ${total_value:.2f}")

    # Main function logic
    if __name__ == "__main__":
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
    RED = "\033[91m"
    BLUE = "\033[38;5;24m"
    #ORANGE = "\033[38;5;214m"
    #GREEN = "\033[38;5;22m"
    #LIGHT_GRAY = "\033[38;5;250m"
    #DARK_GRAY = "\033[38;5;235m"
    BROWN = "\033[38;5;130m"
    RESET = "\033[0m"

    import yfinance as yf
    import mplfinance as mpf
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Dictionary to map numbers to valid periods and intervals accepted by yfinance
    timeframe_options = {
        '1': ('1d', '1m'),    # 1-day chart with 1-minute interval
        '2': ('ytd', '1d'),   # Year-to-date chart with daily interval
        '3': ('1y', '1d'),    # 1-year chart with daily interval
        '4': ('2y', '1d'),    # 2-year chart with daily interval
        '5': ('5y', '1d'),    # 5-year chart with daily interval
        '6': ('10y', '1d')    # Maximum available data
    }

    def plot_stock_chart(ticker, timeframe):
        try:
            # Get the period and interval for the selected option
            period, interval = timeframe_options[timeframe]

            # Download historical stock data with appropriate interval
            stock_data = yf.download(ticker, period=period, interval=interval)

            # Check if data is available
            if stock_data.empty:
                print(f"No data available for {ticker} in the selected timeframe.")
                return

            # Customize the market colors: up candles are dark blue, down candles are black
            mc = mpf.make_marketcolors(up='lime', down='red', wick='inherit', edge='inherit', volume='darkblue')

            # Use the default style but with custom market colors
            s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

            # Create the plot
            fig, axes = mpf.plot(stock_data, type='candle', style=s, title='',  # Set title='' to avoid duplicate title
                                volume=True, volume_panel=1, tight_layout=True, returnfig=True)

            # Set the outer figure background (margins) to black
            fig.patch.set_facecolor('black')

            # Customize x and y labels/titles to be white
            for ax in axes:
                ax.tick_params(axis='x', labelsize=6, colors='white')  # Set x-axis font size to 6 and color to white
                ax.tick_params(axis='y', labelsize=6, colors='white')  # Set y-axis tick label size and color to white
                ax.yaxis.label.set_size(6)                             # Set y-axis label size to 6
                ax.yaxis.label.set_color('white')                      # Set y-axis label color to white

            # Manually set title font size and color to white
            axes[0].set_title(f"{ticker.upper()} - {list(timeframe_options.keys())[int(timeframe) - 1]} Chart", fontsize=8, color='white')

            # Get the latest closing price using .iloc to avoid FutureWarning
            latest_price = stock_data['Close'].iloc[-1]

            # Add latest price in the top-left corner of the main panel (axes[0] is the price panel) with white text
            axes[0].text(0.01, 0.95, f"${latest_price:.2f}", transform=axes[0].transAxes,
                        fontsize=7, color='orange', verticalalignment='top')

            # Adjust candle width manually using matplotlib
            for rect in axes[0].patches:
                if isinstance(rect, Rectangle):
                    rect.set_width(0.75)  # Adjust the width of candles (0.75 is an example value, adjust as needed)

            # Show the plot
            plt.show(block=False)

        except Exception as e:
            print(f"An error occurred: {e}")

    def main():
        # Input from user for stock ticker
        ticker = input("Enter the stock ticker symbol: ").upper()

        # Display options for timeframes
        print(f"{BLUE}Choose period:{RESET}")
        print(f"{RED}[1]{RESET} {BROWN}1d{RESET}")
        print(f"{RED}[2]{RESET} {BROWN}YTD{RESET}")
        print(f"{RED}[3]{RESET} {BROWN}1Y{RESET}")
        print(f"{RED}[4]{RESET} {BROWN}2Y{RESET}")
        print(f"{RED}[5]{RESET} {BROWN}5Y{RESET}")
        print(f"{RED}[6]{RESET} {BROWN}10Y{RESET}")

        # Validate and get user's choice for timeframe
        timeframe = input("Enter your choice (1-6): ")

        # Validate user input for timeframe
        if timeframe not in timeframe_options:
            print(f"Invalid choice '{timeframe}'. Please enter a number between 1 and 6.")
            return

        # Plot the chart with the selected timeframe
        plot_stock_chart(ticker, timeframe)

    # Run the main function if this script is executed
    if __name__ == "__main__":
        main()


    
def cc():
    import yfinance as yf
    from tabulate import tabulate

    # ANSI color codes
    LIME_GREEN = '\033[1;32m'
    NEON_RED = '\033[91m'
    ORANGE = "\033[38;5;130m"  # orange color for commodity names and tickers
    DARK_GRAY = "\033[30m"  # dark gray color for table gridlines
    GRAY = "\033[38;5;250m"
    BLUE = "\033[38;5;24m"  # color for header names
    RESET = '\033[30m'

    # Tickers for Commodities and Cryptocurrencies
    commodity_tickers = {
        "SP500": "VOO",
        "NASDAQ": "QQQ",
        "TQQQ": "TQQQ",
        "QLD": "QLD",
        "NTSX": "NTSX",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "10Y": "^TNX",
        "30Y": "^TYX",
        "Gold": "GC=F"
    }

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def calculate_percent_change(current_price, past_price):
        if past_price == 0:
            return 'N/A'
        return ((current_price - past_price) / past_price) * 100

    def get_data(tickers):
        performance = []
        for name, ticker in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="5y")  # Fetch 5 years of data
                
                if not data.empty:
                    # Latest price and today's data
                    current_price = data['Close'].iloc[-1]
                    today_open_price = data['Open'].iloc[-1]
                    percent_change_today = calculate_percent_change(current_price, today_open_price)
                    
                    # 1 Year data
                    one_year_ago = data.index[-1] - pd.DateOffset(years=1)
                    one_year_price = data.loc[data.index >= one_year_ago, 'Close'].iloc[0] if len(data.loc[data.index >= one_year_ago]) > 0 else current_price
                    percent_change_1y = calculate_percent_change(current_price, one_year_price)
                    
                    # 5 Years data
                    five_years_ago = data.index[0]
                    five_year_price = data.loc[data.index[0], 'Close'] if len(data) > 0 else current_price
                    percent_change_5y = calculate_percent_change(current_price, five_year_price)
                else:
                    current_price = 'N/A'
                    percent_change_today = 'N/A'
                    percent_change_1y = 'N/A'
                    percent_change_5y = 'N/A'
                
                performance.append([
                    f"{ORANGE}{name}{RESET}",
                    f"{ORANGE}{ticker}{RESET}",
                    f"{LIME_GREEN}${current_price:.2f}{RESET}" if isinstance(current_price, float) else f"{LIME_GREEN}{current_price}{RESET}",
                    colorize_percent(percent_change_today) if isinstance(percent_change_today, float) else f"{LIME_GREEN}{percent_change_today}{RESET}",
                    colorize_percent(percent_change_1y) if isinstance(percent_change_1y, float) else f"{LIME_GREEN}{percent_change_1y}{RESET}",
                    colorize_percent(percent_change_5y) if isinstance(percent_change_5y, float) else f"{LIME_GREEN}{percent_change_5y}{RESET}"
                ])
            except Exception as e:
                performance.append([
                    f"{ORANGE}{name}{RESET}",
                    f"{ORANGE}{ticker}{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}"
                ])
                print(f"Error retrieving data for {name}: {e}")
        return performance

    # Get commodity data
    commodity_performance = get_data(commodity_tickers)

    # Print performance table with dark gray gridlines
    headers = [f"{GRAY}Commodity{RESET}", f"{GRAY}Ticker{RESET}", f"{GRAY}Price{RESET}", f"{GRAY}1D %Δ{RESET}", f"{GRAY}1Y %Δ{RESET}", f"{GRAY}5Y %Δ{RESET}"]
    table = tabulate(commodity_performance, headers=headers, tablefmt="grid")

    # Function to color only the structural gridlines
    def color_structural_gridlines(table_string, color):
        lines = table_string.split('\n')
        colored_lines = []
        for i, line in enumerate(lines):
            if i == 0 or i == len(lines) - 1 or set(line) <= set('+-=|'):  # First, last, or separator lines
                colored_line = color + ''.join([char if char in '+-=|' else RESET + char + color for char in line]) + RESET
            else:
                colored_line = color + '|' + RESET + line[1:-1] + color + '|' + RESET
            colored_lines.append(colored_line)
        return '\n'.join(colored_lines)

    # Apply dark gray color to structural gridlines
    colored_table = color_structural_gridlines(table, DARK_GRAY)

    print(colored_table)


def wl():
    import yfinance as yf
    from tabulate import tabulate

    # ANSI color codes
    LIME_GREEN = '\033[1;32m'
    NEON_RED = '\033[91m'
    GRAY = "\033[38;5;250m"
    ORANGE = "\033[38;5;130m"  # orange color for commodity names and tickers
    DARK_GRAY = "\033[30m"  # dark gray color for table gridlines
    BLUE = "\033[38;5;24m"  # color for header names
    RESET = '\033[30m'

    # Tickers for Commodities and Cryptocurrencies
    commodity_tickers = {
        "Interactive Brokers": "IBKR",
        "Zoom Video Communications": "ZM",
        "Amazon.com": "AMZN",
        "NVIDIA": "NVDA",
        "Block": "SQ",
        "Rocket Lab USA": "RKLB",
        "Super Micro Computer": "SMCI"
    }

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def calculate_percent_change(current_price, past_price):
        if past_price == 0:
            return 'N/A'
        return ((current_price - past_price) / past_price) * 100

    def get_data(tickers):
        performance = []
        for name, ticker in tickers.items():
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period="5y")  # Fetch 5 years of data
                
                if not data.empty:
                    # Latest price and today's data
                    current_price = data['Close'].iloc[-1]
                    today_open_price = data['Open'].iloc[-1]
                    percent_change_today = calculate_percent_change(current_price, today_open_price)
                    
                    # 1 Year data
                    one_year_ago = data.index[-1] - pd.DateOffset(years=1)
                    one_year_price = data.loc[data.index >= one_year_ago, 'Close'].iloc[0] if len(data.loc[data.index >= one_year_ago]) > 0 else current_price
                    percent_change_1y = calculate_percent_change(current_price, one_year_price)
                    
                    # 5 Years data
                    five_years_ago = data.index[0]
                    five_year_price = data.loc[data.index[0], 'Close'] if len(data) > 0 else current_price
                    percent_change_5y = calculate_percent_change(current_price, five_year_price)
                else:
                    current_price = 'N/A'
                    percent_change_today = 'N/A'
                    percent_change_1y = 'N/A'
                    percent_change_5y = 'N/A'
                
                performance.append([
                    f"{ORANGE}{name}{RESET}",
                    f"{ORANGE}{ticker}{RESET}",
                    f"{LIME_GREEN}${current_price:.2f}{RESET}" if isinstance(current_price, float) else f"{LIME_GREEN}{current_price}{RESET}",
                    colorize_percent(percent_change_today) if isinstance(percent_change_today, float) else f"{LIME_GREEN}{percent_change_today}{RESET}",
                    colorize_percent(percent_change_1y) if isinstance(percent_change_1y, float) else f"{LIME_GREEN}{percent_change_1y}{RESET}",
                    colorize_percent(percent_change_5y) if isinstance(percent_change_5y, float) else f"{LIME_GREEN}{percent_change_5y}{RESET}"
                ])
            except Exception as e:
                performance.append([
                    f"{ORANGE}{name}{RESET}",
                    f"{ORANGE}{ticker}{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}"
                ])
                print(f"Error retrieving data for {name}: {e}")
        return performance

    # Get commodity data
    commodity_performance = get_data(commodity_tickers)

    # Print performance table with dark gray gridlines
    headers = [f"{GRAY}Company{RESET}", f"{GRAY}Ticker{RESET}", f"{GRAY}Current Price{RESET}", f"{GRAY}1D %Δ{RESET}", f"{GRAY}1Y %Δ{RESET}", f"{GRAY}5Y %Δ{RESET}"]
    table = tabulate(commodity_performance, headers=headers, tablefmt="grid")

    # Function to color only the structural gridlines
    def color_structural_gridlines(table_string, color):
        lines = table_string.split('\n')
        colored_lines = []
        for i, line in enumerate(lines):
            if i == 0 or i == len(lines) - 1 or set(line) <= set('+-=|'):  # First, last, or separator lines
                colored_line = color + ''.join([char if char in '+-=|' else RESET + char + color for char in line]) + RESET
            else:
                colored_line = color + '|' + RESET + line[1:-1] + color + '|' + RESET
            colored_lines.append(colored_line)
        return '\n'.join(colored_lines)

    # Apply dark gray color to structural gridlines
    colored_table = color_structural_gridlines(table, DARK_GRAY)

    print(colored_table)




def des():
    ORANGE = "\033[38;5;130m"
    import yfinance as yf
    from colorama import Fore, Style, init
    import textwrap
    import pandas as pd

    # Initialize colorama
    init(autoreset=True)

    def print_colored_percentage(change, label):
        if change >= 0:
            color = Fore.GREEN
        else:
            color = Fore.RED
        print(f"{ORANGE}{label}: {color}{change:.2f}%{Style.RESET_ALL}")

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
        print(f"\n{ORANGE}Description:{Style.RESET_ALL}\n{textwrap.fill(description, width=80)}")
        print(f"{ORANGE}Price:{Style.RESET_ALL} ${price:.2f}")
        print(f"{ORANGE}Market Cap:{Style.RESET_ALL} ${market_cap / 1e9:.2f} Billion")

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
    ORANGE = "\033[38;5;130m"
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
                first_col_colored = f"{ORANGE}{first_col}{Style.RESET_ALL}"
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

def qu():
    import yfinance as yf
    from datetime import datetime

    # Color definitions
    RED = "\033[91m"
    BROWN = "\033[38;5;130m"
    CYAN = "\033[36m"
    LIGHT_GRAY = "\033[38;5;250m"
    GREEN = '\033[92m'
    RESET = "\033[0m"  # Reset color

    def get_stock_data(ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        history = stock.history(period="5d")
        
        current_price = history['Close'].iloc[-1]
        prev_close = history['Close'].iloc[-2]
        percent_change = ((current_price - prev_close) / prev_close) * 100

        data = {
            'Date': datetime.now().strftime('%d %b'),
            'Ticker': ticker.upper(),
            'Latest Price': f"${current_price:.2f}",
            '1 Day % Change': f"{percent_change:.2f}%",
            'Open': f"${history['Open'].iloc[-1]:.2f}",
            'High': f"${history['High'].iloc[-1]:.2f}",
            'Low': f"${history['Low'].iloc[-1]:.2f}",
            'Close': f"${history['Close'].iloc[-1]:.2f}",
            'Volume': f"{history['Volume'].iloc[-1]:,}",
            'Company Name': info.get('longName', 'N/A'),
            'Market Cap': f"${info.get('marketCap', 0) / 1e9:.2f}B",
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Country': info.get('country', 'N/A'),
            'Dividend Yield': f"{info.get('dividendYield', 0) * 100:.2f}%",
            'Trailing PE': f"{info.get('trailingPE', 'N/A'):.1f}",
            'Forward PE': f"{info.get('forwardPE', 'N/A'):.1f}",
            'Beta': f"{info.get('beta', 'N/A')}",
            'Price to Book': f"{info.get('priceToBook', 'N/A'):.1f}",
            '52 Week High': f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            '52 Week Low': f"${info.get('fiftyTwoWeekLow', 'N/A')}",
            'Enterprise Value': f"${info.get('enterpriseValue', 0) / 1e9:.2f}B",
            'Total Revenue': f"${info.get('totalRevenue', 0) / 1e9:.2f}B",
            'Earnings Growth': f"{info.get('earningsGrowth', 0) * 100:.2f}%"
        }
        return data

    def display_stock_data(data):
        # Conditional color formatting
        change_color = GREEN if float(data['1 Day % Change'][:-1]) > 0 else RED
        p_color = GREEN if float(data['1 Day % Change'][:-1]) > 0 else RED

        print(f"{CYAN}{data['Ticker']}{RESET}  {CYAN}{RESET}{data['Company Name']}  {p_color}{data['Latest Price']}{RESET}  {change_color}{data['1 Day % Change']}{RESET}  {LIGHT_GRAY}{data['Date']}{RESET}")
        print(f"O{BROWN}{data['Open']}{RESET}  H{BROWN}{data['High']}{RESET}  L{BROWN}{data['Low']}{RESET}  C{BROWN}{data['Close']}{RESET}  Vol{BROWN}{data['Volume']}{RESET}  ")
        print(f"{CYAN}MC{RESET}{BROWN}{data['Market Cap']}{RESET}  {CYAN}Ind{RESET}{BROWN}{data['Industry']}{RESET}  {CYAN}{data['Country']}{RESET}")
        print(f"{CYAN}DY{RESET}{BROWN}{data['Dividend Yield']}{RESET}  {CYAN}TPE{RESET}{BROWN}{data['Trailing PE']}{RESET}  {CYAN}FPE{RESET}{BROWN}{data['Forward PE']}{RESET}  {CYAN}Beta{RESET}{BROWN}{data['Beta']}{RESET}  {CYAN}PB{RESET}{BROWN}{data['Price to Book']}{RESET}")
        print(f"{CYAN}52H{RESET}{BROWN}{data['52 Week High']}{RESET}  {CYAN}52L{RESET}{BROWN}{data['52 Week Low']}{RESET}  {CYAN}EV{RESET}{BROWN}{data['Enterprise Value']}{RESET}  {CYAN}REV{RESET}{BROWN}{data['Total Revenue']}{RESET}  {CYAN}EG{RESET}{BROWN}{data['Earnings Growth']}{RESET}")

    if __name__ == "__main__":
        ticker = input("Enter a stock ticker: ")
        stock_data = get_stock_data(ticker)
        display_stock_data(stock_data)


def cl():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import webbrowser
    from tabulate import tabulate

    # ANSI escape codes for colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[38;5;250m'

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
        columns_to_drop = ['1d', '1w', '1m', '6m', 'Company Name', 'FilingDate', 'Industry']  # Add the company name column here
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
    import webbrowser

    # ANSI escape codes for color
    BLUE = "\033[38;5;24m"
    GRAY = "\033[38;5;250m"
    RED = "\033[91m"
    ORANGE = "\033[38;5;130m"
    RESET = "\033[38;5;250m"
    

    def main():
        # List of hedge funds and their corresponding links
        hedge_funds = {
            1: "Scion Asset Management LLC",
            2: "Berkshire Hathaway Inc",
            3: "Icahn Carl C et al",
            4: "Towle & Co",
            5: "Fairfax Financial Holdings Ltd (CAN)",
            6: "Baupost Group LLC (MA)",
            7: "Pamet Capital Management LP",
            8: "Appaloosa Management LP",
            9: "Greenlight Re (David Einhorn)",  # Link is from DataRoma
            10: "Lapides Asset Management LLC",
            11: "Perceptive Advisors LLC",
            12: "Ares Management LLC",
            13: "Oaktree Capital Management LLC",
            14: "MHR Fund Management LLC",
            15: "Hussman Econometrics Advisors Inc",
            16: "Third Avenue Management LLC",
            17: "Walthausen & Co LLC",
            18: "Portolan Capital Management LLC",
            19: "Gavekal Capital LLC",
            20: "Signia Capital Management LLC",
            21: "Bernzott Capital Advisors",
            22: "Sheffield Asset Management",
            23: "Hodges Capital Management Inc",
            24: "Weitz Wallace R & Co",
            25: "Intrepid Capital Management Inc",
            26: "Old West Investment Management LLC",
            27: "Luminus Management LLC",
            28: "O'Shaughnessy Asset Management LLC",
            29: "Kopernik Global Investors LLC", 
            30: "dataroma"
        }

        # Dictionary of links
        links = {
            1: "https://whalewisdom.com/filer/scion-asset-management-llc",
            2: "https://whalewisdom.com/filer/berkshire-hathaway-inc",
            3: "https://whalewisdom.com/filer/icahn-carl-c-et-al",
            4: "https://whalewisdom.com/filer/towle-co",
            5: "https://whalewisdom.com/filer/fairfax-financial-holdings-ltd-can",
            6: "https://whalewisdom.com/filer/baupost-group-llc-ma",
            7: "https://whalewisdom.com/filer/pamet-capital-management-lp",
            8: "https://whalewisdom.com/filer/appaloosa-management-lp",
            9: "https://www.dataroma.com/m/holdings.php?m=GLRE",
            10: "https://whalewisdom.com/filer/lapides-asset-management-llc",
            11: "https://whalewisdom.com/filer/perceptive-advisors-llc",
            12: "https://whalewisdom.com/filer/ares-management-llc",
            13: "https://whalewisdom.com/filer/oaktree-capital-management-llc",
            14: "https://whalewisdom.com/filer/mhr-fund-management-llc",
            15: "https://whalewisdom.com/filer/hussman-econometrics-advisors-inc",
            16: "https://whalewisdom.com/filer/third-avenue-management-llc",
            17: "https://whalewisdom.com/filer/walthausen-amp-co-llc",
            18: "https://whalewisdom.com/filer/portolan-capital-management-llc",
            19: "https://whalewisdom.com/filer/gavekal-capital-llc",
            20: "https://whalewisdom.com/filer/signia-capital-management-llc",
            21: "https://whalewisdom.com/filer/bernzott-capital-advisors",
            22: "https://whalewisdom.com/filer/sheffield-asset-management",
            23: "https://whalewisdom.com/filer/hodges-capital-management-inc",
            24: "https://whalewisdom.com/filer/weitz-wallace-r-co",
            25: "https://whalewisdom.com/filer/intrepid-capital-management-inc",
            26: "https://whalewisdom.com/filer/old-west-investment-management-llc",
            27: "https://whalewisdom.com/filer/luminus-management-llc",
            28: "https://whalewisdom.com/filer/o-shaughnessy-asset-management-llc",
            29: "https://whalewisdom.com/filer/kopernik-global-investors-llc",
            30: "https://www.dataroma.com/m/home.php"

        }

        # Display the list of hedge funds with numbers
        print("Select one or more hedge funds by entering the corresponding numbers (comma-separated), or type 'all' to open all:")
        for number, fund in hedge_funds.items():
            print(f"{RED}[{number}]{RESET}{ORANGE}{fund}{RESET}")

        # Get user input
        choice = input("Enter your choice: ").strip().lower()

        if choice == "all":
            # Open all links
            for link in links.values():
                webbrowser.open(link)
        else:
            # Split the input by commas and try to convert to integers
            try:
                choices = [int(x.strip()) for x in choice.split(",")]
                for num in choices:
                    if num in links:
                        webbrowser.open(links[num])
                    else:
                        print(f"Invalid choice: {num}. Please select a valid number.")
            except ValueError:
                print("Invalid input. Please enter numbers or 'all'.")

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

def gain():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    # Define color codes
    NEON_RED = '\033[91m'
    ORANGE = "\033[38;5;130m"
    RESET = "\033[0m"  # Reset to default color

    def fetch_gainers_table(url):
        # Send a request to the website
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"{ORANGE}Failed to load page {url} with status code {response.status_code}{RESET}")
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Locate the specific table
        table = soup.find('table')
        if table is None:
            raise Exception(f"{ORANGE}Failed to find the table on the page.{RESET}")
        
        # Extract table headers
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        
        # Extract table rows
        rows = []
        for row in table.find_all('tr')[1:]:  # Skip the header row
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            rows.append(cols)
        
        # Create a DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        return df

    def main():
        # Menu for user input
        print(f"{ORANGE}Select an option:{RESET}")
        print(f"{NEON_RED}[1]{RESET} {ORANGE}Daily gainers{RESET}")
        print(f"{NEON_RED}[2]{RESET} {ORANGE}Weekly gainers{RESET}")
        print(f"{NEON_RED}[3]{RESET} {ORANGE}Monthly gainers{RESET}")
        print(f"{NEON_RED}[4]{RESET} {ORANGE}YTD gainers{RESET}")
        print(f"{NEON_RED}[5]{RESET} {ORANGE}1Y gainers{RESET}")
        
        
        choice = input(f"{ORANGE}Enter your choice (1-5): {RESET}")
        
        # Define URLs based on user choice
        urls = {
            '1': "https://stockanalysis.com/markets/gainers/",
            '2': "https://stockanalysis.com/markets/gainers/week/",
            '3': "https://stockanalysis.com/markets/gainers/month/",
            '4': "https://stockanalysis.com/markets/gainers/ytd/",
            '5': "https://stockanalysis.com/markets/gainers/year/"
        }
        
        # Validate choice
        if choice not in urls:
            print(f"{ORANGE}Invalid choice. Please enter a number between 1 and 5.{RESET}")
            return
        
        # Fetch and display the table based on user choice
        url = urls[choice]
        gainers_df = fetch_gainers_table(url)
        
        # Set pandas display options
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being truncated
        
        # Print the DataFrame
        print(f"{ORANGE}{gainers_df}{RESET}")

    if __name__ == "__main__":
        main()

def est():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from io import StringIO
    import webbrowser
    from tabulate import tabulate  # Import tabulate

    # ANSI escape codes for colors
    BROWN = "\033[38;5;130m"
    RED = "\033[91m"
    GRID_COLOR = "\033[30m"  # Dark (black) for grid
    TEXT_COLOR = "\033[0m"  # Default text color
    RESET = "\033[0m"  # Reset color to default

    # Prompt the user to enter a ticker symbol
    ticker = input(f"{BROWN}Enter the ticker symbol (e.g., TSLA): ").strip().upper()

    # Construct the URL using the ticker symbol
    url = f'https://stockanalysis.com/stocks/{ticker}/forecast/'

    # Send a request to fetch the HTML content of the page
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all table elements on the page
    tables = soup.find_all('table')

    # Check if there is at least one table
    if tables:
        # Convert the first table to a string
        table_html = str(tables[0])
        
        # Use StringIO to wrap the HTML string
        table_io = StringIO(table_html)
        
        # Read the HTML table into a DataFrame
        df = pd.read_html(table_io)[0]
        
        # Print the DataFrame using tabulate
        print(f"\n{RED}Analyst Estimates for {RESET}{RED}{ticker}{BROWN}\n")
        
        # Get the tabulated string
        table_str = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
        
        # Replace grid lines with colored versions
        colored_table = table_str.replace('+', f'{GRID_COLOR}+{TEXT_COLOR}') \
                                .replace('|', f'{GRID_COLOR}|{TEXT_COLOR}') \
                                .replace('-', f'{GRID_COLOR}-{TEXT_COLOR}') \
                                .replace('=', f'{GRID_COLOR}-{TEXT_COLOR}')
        
        print(colored_table)  # Print the modified table
    else:
        print(f"{BROWN}No tables found for ticker {ticker}.{RESET}")

    # Ask the user if they want to open the link in a browser
    open_link = input(f"\n{BROWN}Would you like to open the link in your browser? (y/n): {RESET}").strip().lower()

    if open_link == 'y':
        webbrowser.open(url)
    else:
        print(f"{BROWN}The link was not opened.{RESET}")

    
    


def sn():
    import requests
    from bs4 import BeautifulSoup
    import textwrap

    # Color codes
    RED = "\033[91m"
    GRAY = "\033[38;5;250m"
    BROWN = "\033[38;5;130m"
    RESET = "\033[0m"  # Reset to default color

    def wrap_text(text, width=80):
        """Wrap text to specified width, ensuring words are not cut off."""
        return '\n'.join(textwrap.wrap(text, width=width))

    def get_news_text(ticker):
        url = f"https://stockanalysis.com/stocks/{ticker}/"
        response = requests.get(url)
        news_items = []
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            news_divs = soup.find_all('div', class_=['flex flex-col', 'gap-4'])
            
            if not news_divs:
                print(f"{BROWN}No news items found for {ticker}. Please check if the ticker symbol is correct.{RESET}")
                return news_items
            
            for item in news_divs:
                news_item = {}
                
                # Extract title and URL
                title = item.find('h3', class_='mb-2')
                if title:
                    title_link = title.find('a')
                    if title_link:
                        news_item['title'] = title_link.text.strip()
                        news_item['url'] = title_link.get('href')
                    else:
                        news_item['title'] = title.text.strip()
                
                # Extract summary
                summary = item.find('p', class_='overflow-auto')
                if summary:
                    news_item['summary'] = summary.text.strip()
                
                # Extract timestamp and source
                meta_div = item.find('div', class_='mt-1 text-sm text-faded')
                if meta_div:
                    meta_text = meta_div.text.strip()
                    news_item['timestamp'] = meta_div.get('title', meta_text)
                    news_item['source'] = meta_text.split(' - ', 1)[-1] if ' - ' in meta_text else "N/A"
                
                # Extract related symbols if present
                symbols_div = item.find('div', class_='mt-1.5 inline text-light')
                if symbols_div:
                    symbols = [a.text for a in symbols_div.find_all('a', class_='ticker')]
                    if symbols:
                        news_item['related_symbols'] = symbols
                
                if news_item:
                    news_items.append(news_item)
        else:
            print(f"{BROWN}Failed to retrieve the webpage. Status code: {response.status_code}{RESET}")
            print(f"{BROWN}Please check if the ticker symbol is correct.{RESET}")
        
        return news_items

    def print_news_items(news_items):
        seen_titles = set()
        seen_summaries = set()

        for item in news_items:
            # Check if we've seen this title or summary before
            if ('title' in item and item['title'] in seen_titles) or \
            ('summary' in item and item['summary'] in seen_summaries):
                continue  # Skip this item if we've seen it before

            if 'title' in item:
                print(f"{GRAY}{wrap_text(item['title'])}{RESET}")
                seen_titles.add(item['title'])
            if 'summary' in item:
                print(f"{BROWN}{wrap_text(item['summary'])}{RESET}")
                seen_summaries.add(item['summary'])
            if 'timestamp' in item:
                print(f"{BROWN}Timestamp: {item['timestamp']}{RESET}")
            if 'source' in item:
                print(f"{BROWN}Source: {item['source']}{RESET}")
            if 'related_symbols' in item:
                print(f"{BROWN}Related symbols:{RESET}")
                print(f"{BROWN}{wrap_text(', '.join(item['related_symbols']))}{RESET}")
            print(" ")

    def main():
        ticker = input(f"{BROWN}Enter a stock ticker symbol: {RESET}").strip().upper()
        news_items = get_news_text(ticker)
        
        # Reverse the order of news items
        news_items.reverse()
        
        if news_items:
            print_news_items(news_items)
        else:
            print(f"{BROWN}No news found for ticker: {ticker}{RESET}")

    if __name__ == "__main__":
        main()



def mnl(): 
    import os  # Importing os for clearing the terminal
    import requests
    from bs4 import BeautifulSoup
    import textwrap
    import time
    import sys
    import select  # Importing select module for input handling

    # Color codes
    RED = "\033[91m"
    GRAY = "\033[38;5;250m"
    BROWN = "\033[38;5;130m"
    RESET = "\033[0m"  # Reset to default color

    def wrap_text(text, width=80):
        """Wrap text to specified width, ensuring words are not cut off."""
        return '\n'.join(textwrap.wrap(text, width=width))

    def clear_terminal():
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def get_news_text():
        url = "https://stockanalysis.com/news/all-stocks/"
        response = requests.get(url)
        news_items = []
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            news_divs = soup.find_all('div', class_=['flex flex-col', 'gap-4'])
            
            if not news_divs:
                print(f"{BROWN}No news items found at {url}.{RESET}")
                return news_items
            
            for item in news_divs:
                news_item = {}
                
                # Extract title and URL
                title = item.find('h3', class_='mb-2')
                if title:
                    title_link = title.find('a')
                    if title_link:
                        news_item['title'] = title_link.text.strip()
                    else:
                        news_item['title'] = title.text.strip()
                
                # Extract summary
                summary = item.find('p', class_='overflow-auto')
                if summary:
                    news_item['summary'] = summary.text.strip()
                
                # Extract timestamp and source
                meta_div = item.find('div', class_='mt-1 text-sm text-faded')
                if meta_div:
                    meta_text = meta_div.text.strip()
                    news_item['timestamp'] = meta_div.get('title', meta_text)
                    news_item['source'] = meta_text.split(' - ', 1)[-1] if ' - ' in meta_text else "N/A"
                
                # Extract related symbols if present
                symbols_div = item.find('div', class_='mt-1.5 inline text-light')
                if symbols_div:
                    symbols = [a.text for a in symbols_div.find_all('a', class_='ticker')]
                    if symbols:
                        news_item['related_symbols'] = symbols
                
                if news_item:
                    news_items.append(news_item)
        else:
            print(f"{BROWN}Failed to retrieve the webpage. Status code: {response.status_code}{RESET}")
        
        return news_items

    def print_news_items(news_items):
        seen_titles = set()
        seen_summaries = set()

        for item in news_items:
            # Check if we've seen this title or summary before
            if ('title' in item and item['title'] in seen_titles) or \
            ('summary' in item and item['summary'] in seen_summaries):
                continue  # Skip this item if we've seen it before

            if 'title' in item:
                print(f"{GRAY}{wrap_text(item['title'])}{RESET}")
                seen_titles.add(item['title'])
            if 'summary' in item:
                print(f"{BROWN}{wrap_text(item['summary'])}{RESET}")
                seen_summaries.add(item['summary'])
            if 'timestamp' in item:
                print(f"{BROWN}Timestamp: {item['timestamp']}{RESET}")
            if 'source' in item:
                print(f"{BROWN}Source: {item['source']}{RESET}")
            if 'related_symbols' in item:
                print(f"{BROWN}Related symbols:{RESET}")
                print(f"{BROWN}{wrap_text(', '.join(item['related_symbols']))}{RESET}")
            print("\n")

    def countdown(seconds):
        """Display a countdown for the next refresh."""
        for i in range(seconds, 0, -1):
            sys.stdout.write(f"\r{RED}Next refresh for market news in {i} seconds {RESET}")
            sys.stdout.flush()
            time.sleep(1)
            
            # Check for quit command
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input_char = sys.stdin.read(1)
                if input_char.lower() == 'q':
                    print(f"\n{BROWN}Exiting the program. Goodbye!{RESET}")
                    sys.exit(0)
                    
        sys.stdout.write("\r")  # Clears the line after countdown finishes

    def main():
        while True:
            clear_terminal()  # Clear the terminal at the start of each loop
            news_items = get_news_text()
            
            # Reverse the order of news items
            news_items.reverse()
            
            print_news_items(news_items)
            
            # Countdown before next refresh (30 seconds)
            countdown(300)

    if __name__ == "__main__":
        main()



def ipo():
    import pandas as pd
    import requests
    from bs4 import BeautifulSoup
    from tabulate import tabulate

    # ANSI color code for brown
    BROWN = "\033[38;5;130m"
    RESET = "\033[0m"  # Reset to default color

    # URL of the page to scrape
    url = "https://stockanalysis.com/ipos/calendar/"

    # Send a GET request to the page
    response = requests.get(url)

    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table on the page
    table = soup.find('table')

    # Extract headers
    headers = [th.text.strip() for th in table.find('thead').find_all('th')]

    # Extract rows of the table
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        rows.append(cells)

    # Create a DataFrame for better presentation
    df = pd.DataFrame(rows, columns=headers)

    # Apply color to the DataFrame headers and output using tabulate
    colored_output = tabulate(df, headers=[BROWN + h + BROWN for h in df.columns], tablefmt='plain', stralign='left', numalign='right')

    # Print the colored table
    print(colored_output)


def rtc():
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import mplfinance as mpf
    import pandas as pd

    # Define the function to fetch and update data
    def fetch_data(ticker):
        df = yf.download(tickers=ticker, period='1d', interval='1m')
        return df

    # Define the function to update the plot
    def update_plot(frame):
        df = fetch_data(ticker)

        # Clear the existing chart without affecting the text
        ax[0].clear()

        # Plot the updated candlestick chart
        mpf.plot(df, type='candle', ax=ax[0], style=custom_style, volume=False, show_nontrading=False)

        # Update the latest price in the top left
        latest_price = df['Close'].iloc[-1]
        ax[0].text(0.05, 0.95, f'   ${latest_price:.2f}', transform=ax[0].transAxes,
                fontsize=10, verticalalignment='top', color='orange')

        # Set title and labels
        ax[0].set_title(f'{ticker} Live Price', color='grey', fontsize = 10)
        ax[0].set_xlabel('Time', color='grey')
        ax[0].set_ylabel('Price ($)', color='grey')

        # Set smaller font size for x-axis and y-axis labels
        for label in ax[0].get_xticklabels():
            label.set_fontsize(8)
            label.set_color('grey')
        for label in ax[0].get_yticklabels():
            label.set_fontsize(8)
            label.set_color('grey')

        # Set axis grid, colors, and style
        ax[0].grid(True, color='grey', linestyle='--')
        ax[0].tick_params(axis='both', colors='grey')

    # Get ticker input
    ticker = input("Enter a stock ticker symbol: ").upper()

    # Define custom style with lime up candles and red down candles
    custom_market_colors = mpf.make_marketcolors(
        up='lime', down='red', 
        edge='inherit', wick='inherit', volume='in'
    )
    custom_style = mpf.make_mpf_style(
        base_mpl_style='dark_background', marketcolors=custom_market_colors
    )

    # Set up the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor='black', subplot_kw={'facecolor':'black'})
    ax = [ax]  # Wrap the ax in a list for `mpf.plot` compatibility

    # Create the animation
    ani = animation.FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)

    # Set up plot appearance
    fig.patch.set_facecolor('black')  # Set figure background color
    ax[0].set_facecolor('black')      # Set axis background color

    # Display the chart
    plt.show()

def snl():
    import requests
    from bs4 import BeautifulSoup
    import textwrap
    from datetime import datetime, timedelta
    import pytz
    import re
    from dateutil import parser
    import warnings
    from dateutil.parser import UnknownTimezoneWarning
    import os
    import time

    # Suppress the UnknownTimezoneWarning from dateutil
    warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

    # Color codes
    RED = "\033[91m"
    BLUE = "\033[38;5;24m"
    BROWN = "\033[38;5;130m"
    GRAY = "\033[38;5;250m"
    RESET = "\033[0m"  # Reset to default color

    def wrap_text(text, width=80):
        """Wrap text to specified width, ensuring words are not cut off."""
        return '\n'.join(textwrap.wrap(text, width=width))

    def get_news_text(ticker):
        url = f"https://stockanalysis.com/stocks/{ticker}/"
        response = requests.get(url)
        news_items = []
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            news_divs = soup.find_all('div', class_=['flex flex-col', 'gap-4'])
            
            if not news_divs:
                print(f"{BROWN}No news items found for {ticker}. Please check if the ticker symbol is correct.{RESET}")
                return news_items
            
            for item in news_divs:
                news_item = {}
                
                title = item.find('a', class_='text-default')
                if title:
                    news_item['title'] = title.text.strip()
                
                summary = item.find('p', class_='overflow-auto')
                if summary:
                    news_item['summary'] = summary.text.strip()
                
                meta_div = item.find('div', class_='mt-1')
                if meta_div:
                    news_item['full_timestamp'] = meta_div.get('title', '')
                    news_item['relative_time'] = meta_div.text.strip()
                    
                    source = item.find('a', class_='text-faded')
                    if source:
                        news_item['source'] = source.text.strip()
                
                news_item['ticker'] = ticker
                
                if news_item:
                    news_items.append(news_item)
        else:
            print(f"{BROWN}Failed to retrieve the webpage for {ticker}. Status code: {response.status_code}{RESET}")
        
        return news_items

    def parse_timestamp(timestamp):
        try:
            # Try to parse the timestamp using dateutil
            dt = parser.parse(timestamp)
            
            # If no timezone info, assume US/Eastern
            if dt.tzinfo is None:
                dt = pytz.timezone('US/Eastern').localize(dt)
            
            return dt.astimezone(pytz.UTC)
        except ValueError:
            # If dateutil fails, fall back to relative time parsing
            return parse_relative_time(timestamp)

    def parse_relative_time(timestamp):
        now = datetime.now(pytz.UTC)
        relative_time_match = re.search(r'(\d+)\s+(\w+)\s+ago', timestamp)
        if relative_time_match:
            number, unit = relative_time_match.groups()
            number = int(number)
            if 'minute' in unit:
                return now - timedelta(minutes=number)
            elif 'hour' in unit:
                return now - timedelta(hours=number)
            elif 'day' in unit:
                return now - timedelta(days=number)
            elif 'week' in unit:
                return now - timedelta(weeks=number)
            elif 'month' in unit:
                return now - timedelta(days=number*30)  # Approximation
            elif 'year' in unit:
                return now - timedelta(days=number*365)  # Approximation
        
        # If all parsing attempts fail, return a very old date to sort it at the end
        return datetime.min.replace(tzinfo=pytz.UTC)

    def print_news_items(news_items):
        seen_titles = set()
        seen_summaries = set()

        for item in news_items:
            if ('title' in item and item['title'] in seen_titles) or \
            ('summary' in item and item['summary'] in seen_summaries):
                continue

            if 'title' in item:
                print(f"{GRAY}{wrap_text(item['title'])}{RESET}")
                seen_titles.add(item['title'])
            if 'summary' in item:
                print(f"{BROWN}{wrap_text(item['summary'])}{RESET}")
                seen_summaries.add(item['summary'])
            if 'full_timestamp' in item:
                print(f"{GRAY}{item['relative_time']} {item['ticker']}")
            if 'source' in item:
                print(f"{BROWN}Source: {item['source']}{RESET}")
            print(" ")
            print(" ")

    def main():
        tickers = input(f"{BROWN}Enter stock ticker symbol(s) separated by commas (or 'quit' to exit): {RESET}").strip().upper()
        if tickers.lower() == 'quit':
            print(f"{BROWN}Exiting the program. Goodbye!{RESET}")
            return
        
        while True:
            os.system("clear")  # Clear the terminal on macOS

            all_news_items = []
            for ticker in tickers.split(','):
                ticker = ticker.strip()
                news_items = get_news_text(ticker)
                all_news_items.extend(news_items)
            
            # Sort all news items collectively by timestamp, from oldest to newest
            all_news_items.sort(key=lambda x: parse_timestamp(x['full_timestamp']))
            
            print_news_items(all_news_items)
            
            # Countdown for the next refresh (300 seconds = 5 minutes)
            for remaining in range(300, 0, -1):
                print(f"\r{RED}Refreshing for new stock news in {remaining} seconds...{RESET}", end="")
                time.sleep(1)

    if __name__ == "__main__":
        main()

def qm():
    import yfinance as yf
    from tabulate import tabulate
    import os
    import time
    import pandas as pd

    # ANSI color codes
    LIME_GREEN = '\033[1;32m'
    NEON_RED = '\033[91m'
    GRAY = "\033[38;5;250m"
    ORANGE = "\033[38;5;130m"
    DARK_GRAY = "\033[30m"
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def calculate_percent_change(current_price, past_price):
        if past_price == 0:
            return 'N/A'
        return ((current_price - past_price) / past_price) * 100

    def get_data(tickers):
        performance = []
        for ticker in tickers:
            try:
                # Fetch 1-minute intraday data for the last day
                data = yf.Ticker(ticker).history(period='1d', interval='1m')

                if not data.empty:
                    current_price = data['Close'].iloc[-1]
                    today_open_price = data['Open'].iloc[-1]
                    percent_change_today = calculate_percent_change(current_price, today_open_price)
                    volume = data['Volume'].iloc[-1]  # Get the most recent volume

                    performance.append([
                        f"{ORANGE}{ticker}{RESET}",
                        f"{LIME_GREEN}${current_price:.2f}{RESET}",
                        colorize_percent(percent_change_today),
                        f"{LIME_GREEN}{volume}{RESET}"
                    ])
                else:
                    performance.append([
                        f"{ORANGE}{ticker}{RESET}",
                        f"{LIME_GREEN}N/A{RESET}",
                        f"{LIME_GREEN}N/A{RESET}",
                        f"{LIME_GREEN}N/A{RESET}"
                    ])
            except Exception as e:
                performance.append([
                    f"{ORANGE}Unknown{RESET}",
                    f"{ORANGE}{ticker}{RESET}",
                    f"{LIME_GREEN}N/A{RESET}",
                    f"{LIME_GREEN}N/A{RESET}"
                ])
                print(f"Error retrieving data for {ticker}: {e}")
        return performance

    def color_structural_gridlines(table_string, color):
        lines = table_string.split('\n')
        colored_lines = []
        for i, line in enumerate(lines):
            if i == 0 or i == len(lines) - 1 or set(line) <= set('+-=|'):
                colored_line = color + line.replace('|', f'{color}|{RESET}') + RESET
            else:
                colored_line = color + ''.join(
                    char if char != '|' else f'{color}|{RESET}' for char in line
                )
            colored_lines.append(colored_line + RESET)
        return '\n'.join(colored_lines)

    def main():
        # Ask the user for input tickers
        user_input = input("Enter tickers (1 = default, 2 = general): ")
        
        if user_input.strip() == "1":
            tickers = ['TSLA', 'INTC', 'ASTS', 'SOFI', 'NVTS', 'QQQ', 'TQQQ', 'BTC-USD']
        elif user_input.strip() == "2":
            tickers = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'LLY', 'AVGO', 
                    'TSLA', 'JPM', 'UNH', 'XOM', 'V', 'PG', 'MA', 'COST', 'JNJ', 'HD', 
                    'ABBV', 'WMT', 'NFLX', 'MRK', 'BAC', 'KO', 'ORCL', 'CRM', 'AMD', 'ADBE']
        else:
            # Parse the input into a list
            tickers = [ticker.strip() for ticker in user_input.split(',')]

        refresh_interval = 60  # Refresh every 60 seconds

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            performance = get_data(tickers)

            headers = [f"{BOLD}{ORANGE}Ticker{RESET}", f"{BOLD}{ORANGE}${RESET}", f"{BOLD}{ORANGE}1D %Δ{RESET}", f"{BOLD}{ORANGE}Volume{RESET}"]
            
            # Custom format for tighter spacing
            custom_format = 'grid'
            column_alignments = ['left', 'right', 'right', 'right']
            
            table = tabulate(performance, headers=headers, tablefmt=custom_format, colalign=column_alignments, numalign='decimal')
            colored_table = color_structural_gridlines(table, DARK_GRAY)

            print(colored_table)

            for remaining in range(refresh_interval, 0, -1):
                print(f"\rNext refresh in {remaining} seconds...", end='')
                time.sleep(1)

            print()  # Move to the next line after countdown

    if __name__ == "__main__":
        main()

def si():
    import requests
    from bs4 import BeautifulSoup
    from tabulate import tabulate

    # ANSI escape codes
    BLACK = '\033[0m'
    ORANGE = "\033[38;5;130m"

    # URL to scrape
    url = "https://www.highshortinterest.com/"

    # Fetch the page content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the page
    page_text = soup.get_text(separator="\n", strip=True)

    # Find the position of the word "Ticker" and extract everything after it
    start_position = page_text.find("Ticker")

    # Check if the word "Ticker" is found, then extract text after it
    if start_position != -1:
        page_text_after_ticker = page_text[start_position:]
    else:
        print("The word 'Ticker' was not found in the page.")
        exit()

    # Split the extracted text into lines
    lines = page_text_after_ticker.split("\n")

    # Create a list to store the rows for the table
    table_data = []
    headers = ["Ticker", "Company", "Exchange", "ShortInt", "Float", "Outstd", "Industry"]

    # Remove duplicate headers
    lines = [line for line in lines if line not in headers]

    # Loop through the lines and extract rows
    for i in range(0, len(lines), 7):
        row = lines[i:i+7]
        if len(row) == 7:  # Ensure that we have a complete row
            table_data.append(row)

    # Format the table with colors
    formatted_table = []
    formatted_headers = [ORANGE + header + BLACK for header in headers]
    formatted_table.append(formatted_headers)

    for row in table_data:
        formatted_row = [ORANGE + str(item) + BLACK for item in row]
        formatted_table.append(formatted_row)

    # Create the table with tabulate
    table = tabulate(formatted_table, tablefmt="plain")

    # Replace grid characters with black
    table = table.replace('+', BLACK + '+')\
                .replace('-', BLACK + '-')\
                .replace('|', BLACK + '|')

    # Print the formatted table
    print(BLACK + table + BLACK)


def main():
    while True:
        RED = "\033[91m"
        BLUE = "\033[38;5;24m"
        ORANGE = "\033[38;5;214m"
        GREEN = "\033[38;5;22m"
        LIGHT_GRAY = "\033[38;5;250m"
        DARK_GRAY = "\033[38;5;235m"
        BROWN = "\033[38;5;130m"
        CYAN = "\033[36m"
        RESET = "\033[0m"

        print(f"\n{RED}Terminal {RESET}")
        print(f"{CYAN}pulse:{RESET} {BROWN}[news] [cc] [gm] [wl] [wln] [pn] [cl] [gain] [ipo] [si] [qu]{RESET}")
        print(f"{CYAN}read:{RESET} {BROWN}[sa] [vic] [wsj] [nyt] [brns] [sema] [ft] [sn]{RESET}")
        print(f"{CYAN}research:{RESET} {BROWN}[screen] [fund] [sec] [10k] [10q] [fs] [sta] [roic] [ins] [hol]{RESET}")
        print(f"{CYAN}tools:{RESET} {BROWN}[dcf] [val] [pch] [sc] [ovs] [sim] [op] [port] [est] [des] [ch] [note]{RESET}")
        print(f"{CYAN}live:{RESET} {BROWN}[rtc] [mnl] [snl] [qm]")

        print(f"{LIGHT_GRAY} ---- ")

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
        elif choice == 'sta':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://stockanalysis.com/stocks/{ticker}"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'port':
            port()
        elif choice == 'des':
            des()
        elif choice == 'wln':
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
        elif choice == 'wl':
            wl()
        elif choice == 'screen':
            import webbrowser
            webbrowser.open("https://finviz.com/screener.ashx")
        elif choice == 'pn':
            import webbrowser
            webbrowser.open("https://finviz.com/portfolio.ashx?v=1&pid=1911250")
        elif choice == 'sec':
            sec()
        elif choice == 'wsj':
            import webbrowser
            webbrowser.open("https://www.wsj.com/")
        elif choice == 'nyt':
            import webbrowser
            webbrowser.open("https://www.nytimes.com/")
        elif choice == 'brns':
            import webbrowser
            webbrowser.open("https://www.barrons.com/")
        elif choice == 'sema':
            import webbrowser
            webbrowser.open("https://www.semafor.com/")
        elif choice == 'ft':
            import webbrowser
            webbrowser.open("https://www.ft.com/")
        elif choice == 'note':
            import webbrowser
            webbrowser.open("https://www.rapidtables.com/tools/notepad.html")
        elif choice == 'gain':
            gain()
        elif choice == 'est':
            est()
        elif choice == 'sn':
            sn()
        elif choice == 'mnl':
            mnl()
        elif choice == 'snl':
            snl()
        elif choice == 'qm':
            qm()
        elif choice == 'ipo':
            ipo()
        elif choice == 'rtc':
            rtc()
        elif choice == 'si':
            si()
        elif choice == 'qu':
            qu()
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
