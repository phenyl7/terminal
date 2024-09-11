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

def plot_stock_price(ticker, years, sma_period):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)  # Approximate number of days for given years

    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No data found for ticker: {ticker}")
        return

    if sma_period > 0:
        stock_data['SMA'] = stock_data['Adj Close'].rolling(window=sma_period).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label=f'{ticker} Price', color='orange')
    
    if sma_period > 0:
        plt.plot(stock_data.index, stock_data['SMA'], label=f'SMA {sma_period}', color='blue')

    plt.title(f'{ticker} Stock Price Over the Last {years} Years', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Adjusted Close Price', color='white')
    plt.legend()
    
    plt.gca().set_facecolor('black')  # Background color of the plot
    plt.gca().tick_params(axis='both', colors='grey')  # Color of the ticks
    plt.grid(color='grey', linestyle='--', linewidth=0.5)  # Grid color and style
    
    plt.gcf().patch.set_facecolor('black')  # Background color of the figure
    plt.show(block=False)


def plot_portfolio_performance_chart(years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    num_stocks = len([ticker for ticker in portfolio if ticker != "1"])
    stocks_per_page = 4
    num_pages = (num_stocks + stocks_per_page - 1) // stocks_per_page

    print(f"Total pages: {num_pages}")

    # Iterate through pages
    for page in range(num_pages):
        plt.figure(figsize=(12, 12), facecolor='black')
        
        start_idx = page * stocks_per_page
        end_idx = min(start_idx + stocks_per_page, num_stocks)
        tickers_to_plot = [ticker for ticker in portfolio if ticker != "1"][start_idx:end_idx]

        total_value = 0
        subplot_index = 1

        for ticker in tickers_to_plot:
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)
                
                if stock_data.empty:
                    print(f"No data found for ticker: {ticker}")
                    continue

                ax = plt.subplot(stocks_per_page, 1, subplot_index)
                ax.plot(stock_data.index, stock_data['Adj Close'], label=f'{ticker} Price', color='orange')
                ax.set_title(f'{ticker} Price Over Time', color='white')
                ax.set_ylabel('Adjusted Close Price', color='white')
                ax.legend()
                ax.set_facecolor('black')
                ax.tick_params(axis='both', colors='grey')
                ax.grid(color='grey', linestyle='--', linewidth=0.5)

                subplot_index += 1
                
                last_close_price = stock_data['Close'].iloc[-1]
                total_value += portfolio[ticker]['shares'] * last_close_price
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        plt.subplots_adjust(hspace=0.5)
        plt.figtext(0.1, 0.02, f'Total Portfolio Value (Page {page + 1}/{num_pages}): ${total_value:,.2f}', color='orange', fontsize=12)
        
        plt.show(block=False)
        
        if page < num_pages - 1:
            input("Press Enter to continue to the next page...")


def gm():
    import json
    import yfinance as yf
    import webbrowser
    from datetime import datetime

    # Define ANSI color codes
    YELLOW = '\033[33m'
    LIME_GREEN = '\033[92m'
    NEON_RED = '\033[91m'
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

    def open_websites():
        urls = [
            "https://x.com",
            "https://www.reddit.com/r/wallstreetbets/",
            "https://news.google.com/home"
        ]
        
        for url in urls:
            webbrowser.open_new_tab(url)

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

    if __name__ == "__main__":
        print(get_greeting())
        
        portfolio_file = 'portfolio.json'  # Path to your portfolio.json file
        portfolio = load_portfolio(portfolio_file)
        performance, total_value = calculate_performance(portfolio)
        print_performance(performance, total_value)
        
        # Prompt for opening websites
        response = input("Open x, wsb, and google news? (y/n): ").strip().lower()
        if response == 'y':
            open_websites()


def add_position():
    ticker = input("Enter ticker: ")
    cost_basis = float(input("Enter cost basis: "))
    shares = float(input("Enter number of shares: "))
    portfolio[ticker] = {'shares': shares, 'cost_basis': cost_basis}
    save_portfolio(portfolio)
    print(f"Added {ticker} to portfolio.")

def news():
    import yfinance as yf
    import webbrowser

    # ANSI escape codes for colors
    YELLOW = '\033[33m'    # Yellow text
    WHITE = '\033[97m'     # White text
    RESET = '\033[0m'      # Reset to default color

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

    def display_headlines(ticker, headlines, start_number):
        for idx, (title, _, publisher) in enumerate(headlines, start=start_number):
            print(f"[{idx}] {ticker} - {YELLOW}Title: {title}{RESET}")
            print(f"    {WHITE}Publisher: {publisher}{RESET}")
            print()

    def search_links_on_google(links):
        for link in links:
            webbrowser.open_new_tab(link)

    # Get ticker symbols from the user, separated by commas
    ticker_symbols = input("Enter ticker symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ")
    ticker_symbols = [symbol.strip() for symbol in ticker_symbols.split(',')]

    all_headlines = []
    start_number = 1
    for symbol in ticker_symbols:
        print(f"\nFetching news for {symbol}...")
        headlines = fetch_news(symbol)
        all_headlines.extend((symbol, title, link) for title, link, publisher in headlines)
        # Display headlines with numbering and stock ticker
        display_headlines(symbol, headlines, start_number)
        start_number += len(headlines)

    # Prompt the user to search specific titles
    while True:
        try:
            numbers_input = input("Enter the numbers of the titles you want to search, separated by commas (e.g., 1,3,5): ")
            numbers = [int(num.strip()) for num in numbers_input.split(',')]

            # Validate and search links
            if all(1 <= number <= len(all_headlines) for number in numbers):
                links = [all_headlines[number - 1][2] for number in numbers]
                search_links_on_google(links)
                break
            else:
                print(f"Invalid numbers. Please enter numbers between 1 and {len(all_headlines)}.")
        except ValueError:
            print("Invalid input. Please enter valid numbers separated by commas.")

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
    

def remove_position():
    ticker = input("Enter ticker to remove: ")
    if ticker in portfolio:
        del portfolio[ticker]
        save_portfolio(portfolio)
        print(f"Removed {ticker} from portfolio.")
    else:
        print("Ticker not found in portfolio.")

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
        "Silver": "SI=F",
        "Copper": "HG=F",
        "WTI Crude Oil": "CL=F",
        "Natural Gas": "NG=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F",
        "Lumber": "LBR=F"
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


def main():
    while True:
        print("\nMenu:")
        print("[1] Chart             [2] News")
        print("[3] Markets           [4] Good Morning")
        print("[5] SA                [6] Finviz")
        print("[7] 10-K              [8] 10-Q")
        print("[9] StockCharts       [10] Insiders")
        print("[11] Quote            [12] Financials")
        print("[13] Ratios           [14] Portfolio Vis")
        print("[15] New Entry        [16] Edit Port")
        print("[17] DES              [18] Finviz News")
        print("[19] Options          [20] Simulations")
        print("[21] OVS              [q] Exit")

        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            ticker = input("Enter ticker: ").strip().upper()
            try:
                years = float(input("Enter years: ").strip())
                if years <= 0:
                    raise ValueError("The timespan must be a positive number.")
                sma_period = int(input("Enter SMA period (0 to skip): ").strip())
                if sma_period < 0:
                    raise ValueError("The SMA period must be a non-negative integer.")
            except ValueError as e:
                print(f"Invalid input: {e}")
            else:
                plot_stock_price(ticker, years, sma_period)
        elif choice == '2':
            news()
        elif choice == '3':
            cc()
        elif choice == '4':
            gm()
        elif choice == '5':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://seekingalpha.com/symbol/{ticker}"
            webbrowser.open(url)
        elif choice == '6':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
            webbrowser.open(url)
        elif choice == '7':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-K"
            webbrowser.open(url)
        elif choice == '8':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-Q"
            webbrowser.open(url)
        elif choice == '9':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://stockcharts.com/sc3/ui/?s={ticker}"
            webbrowser.open(url)
        elif choice == '10':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"http://openinsider.com/search?q={ticker}"
            webbrowser.open(url)
        elif choice == '11':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}/financials"
            webbrowser.open(url)
        elif choice == '12':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}"
            webbrowser.open(url)
        elif choice == '13':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}/ratios"
            webbrowser.open(url)
        elif choice == '14':
            try:
                years = int(input("Enter number of years for portfolio performance chart: ").strip())
                if years <= 0:
                    raise ValueError("The number of years must be a positive integer.")
            except ValueError as e:
                print(f"Invalid input: {e}")
            else:
                plot_portfolio_performance_chart(years)
        elif choice == '15':
            add_position()
        elif choice == '16':
            remove_position()
        elif choice == '17':
            des()
        elif choice == '18':
            webbrowser.open("https://finviz.com/news.ashx")
        elif choice == '19':
            options()
        elif choice == '20':
            sim()
        elif choice == '21':
            ovs()
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
