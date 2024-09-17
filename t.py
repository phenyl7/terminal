import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import webbrowser 

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
    from datetime import datetime, timedelta
    import re

    def parse_duration(duration):
        """Convert duration string to number of days."""
        match = re.match(r"(\d+)([dmy])", duration.strip().lower())
        if not match:
            print("Invalid duration format. Use 'd' for days, 'm' for months, 'y' for years.")
            return None
        
        number, unit = match.groups()
        number = int(number)
        
        if unit == 'd':
            return number
        elif unit == 'm':
            return number * 30  # Approximate number of days in a month
        elif unit == 'y':
            return number * 365.25  # Approximate number of days in a year
        else:
            print("Unsupported unit. Use 'd' for days, 'm' for months, 'y' for years.")
            return None

    def plot_stock_price():
        # Prompt for ticker and duration
        ticker = input("Enter the stock ticker: ")
        duration_input = input("Enter the duration (e.g., 1d for 1 day, 7d for 7 days, 1m for 1 month, 2y for 2 years): ")

        # Convert duration input to number of days
        days = parse_duration(duration_input)
        if days is None:
            return

        # Download stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
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

        # Dynamic adjustment for short timeframes
        if days <= 30:  # For timeframes up to 30 days
            ax1[0].set_ylim(y_min * 0.98, y_max * 1.02)  # Slightly expanded range for better visibility
        elif days <= 365:  # For timeframes up to 1 year
            ax1[0].set_ylim(y_min * 0.95, y_max * 1.05)
        else:  # For timeframes longer than 1 year
            ax1[0].set_ylim(y_min * 0.90, y_max * 1.10)

        ax1[0].set_title(f'{ticker} {duration_input}', color='white')
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

    def fetch_news(ticker_symbol):
        ticker = yf.Ticker(ticker_symbol)
        news = ticker.news

        headlines = []
        for article in news[:8]:  # Always fetch 8 headlines
            title = article.get('title', 'No title available')
            link = article.get('link', 'No link available')
            publisher = article.get('publisher', 'No publisher available')
            headlines.append((title, link, publisher))
            
        return headlines

    def display_all_headlines(news_data):
        counter = 1  # Unique number for each headline
        headline_map = {}  # Store all headlines with numbers
        for ticker, headlines in news_data.items():
            print(f"\n{YELLOW}News for {ticker.upper()}:{RESET}")
            for title, link, publisher in headlines:
                print(f"[{counter}] {YELLOW}Title: {title}{RESET}")
                print(f"    {WHITE}Publisher: {publisher}{RESET}")
                headline_map[counter] = link
                counter += 1
        return headline_map

    def open_link_in_browser(headline_map):
        try:
            choice = int(input("Enter the number of the article to open (or 0 to skip): "))
            if choice in headline_map:
                link = headline_map[choice]
                print(f"Opening article...")
                webbrowser.open(link)
            elif choice == 0:
                print("Skipping article opening...")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")

    # Main function logic
    print(get_greeting())
    
    portfolio_file = 'portfolio.json'  # Path to your portfolio.json file
    portfolio = load_portfolio(portfolio_file)
    performance, total_value = calculate_performance(portfolio)
    print_performance(performance, total_value)

    # Fetch and display news for each ticker
    news_data = {}
    for ticker in portfolio.keys():
        if ticker == "1":
            continue
        news_data[ticker] = fetch_news(ticker)

    # Display all headlines with unique numbers and map them for opening
    headline_map = display_all_headlines(news_data)

    # Prompt to open a link in the browser
    open_link_in_browser(headline_map)


def news():
    import feedparser
    import textwrap
    import os
    import sys
    import time

    # List of RSS feed URLs
    RSS_FEEDS = [
        'https://feeds.content.dowjones.io/public/rss/mw_topstories',
        'https://rss.nytimes.com/services/xml/rss/nyt/US.xml',
        'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml'
        'https://rss.nytimes.com/services/xml/rss/nyt/World.xml'
        
    ]

    # Define maximum width for text wrapping
    MAX_WIDTH = 80

    # Define refresh interval in seconds (e.g., 5 minutes)
    REFRESH_INTERVAL = 300

    # ANSI escape codes for colors
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    def print_wrapped(text, width=MAX_WIDTH):
        """Print text with word wrapping."""
        wrapped_text = textwrap.fill(text, width=width)
        print(wrapped_text)

    def clear_screen():
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def fetch_rss_feed(url):
        feed = feedparser.parse(url)
        # Print feed title in red
        print(f"\n{'='*MAX_WIDTH}\n{RED}Feed Title: {feed.feed.title}{RESET}\n{'='*MAX_WIDTH}")
        
        for entry in feed.entries:
            # Extract relevant information
            title = entry.title
            summary = entry.summary if 'summary' in entry else None
            
            # Print headline in blue and summary if available
            if summary:
                # Print headline in blue
                print_wrapped(f"{BLUE}{title}{RESET}")
                
                # Add two lines of space
                print()
                
                # Print summary
                print_wrapped(f"   Summary: {summary}")
                
                # Add a line of space between each headline
                print()

    def main():
            
        print("\nFetching the latest RSS feed items...\n")
            
        for url in RSS_FEEDS:
            print(f"Fetching from {url}...\n")
            fetch_rss_feed(url)
            
    if __name__ == "__main__":
        main()



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
    import datetime
    from mplchart.chart import Chart
    from mplchart.primitives import Candlesticks, Volume
    from mplchart.indicators import ROC, SMA, EMA, RSI, MACD

    # User input for ticker and number of years
    ticker = input("Enter stock ticker: ")
    years = float(input("Enter number of years of data: "))

    # Calculate the start and end dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=int(years * 365.25))  # Using 365.25 to account for leap years

    # Fetch historical data
    prices = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate percent change
    start_price = prices['Close'].iloc[0]  # Price at the start date
    end_price = prices['Close'].iloc[-1]   # Price at the end date
    percent_change = ((end_price - start_price) / start_price) * 100

    # Define the number of bars to display (max_bars) - can be adjusted as needed
    max_bars = 250

    # Define indicators
    indicators = [
        Candlesticks(),  
        Volume(),
        RSI(),
        MACD()
    ]

    # Create and plot the chart with percent change in the title
    chart = Chart(title=f'{ticker} - {percent_change:.2f}% Change', max_bars=max_bars)
    chart.plot(prices, indicators)
    chart.show()

    
def cc():
    import yfinance as yf

    # ANSI color codes
    LIME_GREEN = '\033[92m'
    NEON_RED = '\033[91m'
    BLUE = '\033[94m'  # Blue color for commodity names
    RESET = '\033[0m'

    # Tickers for Commodities and Cryptocurrencies
    commodity_tickers = {
        "Gold": "GC=F",
       
    }

    # Tickers for Cryptocurrencies and Indices
    crypto_tickers = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD"
    }

    index_tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ 100": "^NDX",
    }

    def colorize_percent(percent):
        if isinstance(percent, (int, float)):
            return f"{LIME_GREEN}{percent:.2f}%{RESET}" if percent >= 0 else f"{NEON_RED}{percent:.2f}%{RESET}"
        return percent

    def print_performance(performance):
        for name, data in performance.items():
            print(f"{BLUE}Commodity: {name} ({data['ticker']}){RESET}")
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

    # Get and print cryptocurrency data
    crypto_performance = get_data(crypto_tickers)
    print_performance(crypto_performance)

    # Get and print index data
    index_performance = get_data(index_tickers)
    print_performance(index_performance)

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



def main():
    while True:
        print("\nMenu:")
        print("[ch]   [news] [cc]")
        print("[gm]   [des]  [fvn]")
        print("[sa]   [fv]   [hol] ")
        print("[ins]  [roic] [fs] ")
        print("[port] [10k]  [10q]  ")
        print("[op]   [sim]  [ovs]")
        print("[vic]  [gain] [val]")
        print("[dcf]  [sc]   [q]   ")
     
        
        choice = input("Choose an option: ").strip()
        
        if choice == 'ch':
            plot()
        elif choice == 'news':
            news()
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
        elif choice == 'fvn':
            import webbrowser
            webbrowser.open("https://finviz.com/news.ashx")
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
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
