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



def plot_stock_price():
    # Prompt for ticker and years
    ticker = input("Enter the stock ticker: ")
    years = int(input("Enter the number of years of data to retrieve: "))

    # Define the plot types and ask the user to select one
    print("Select plot type:")
    print("1. SMA")
    print("2. Bollinger Bands")
    print("3. RSI")
    print("4. MACD")
    print("5. Ichimoku Cloud")

    plot_choice = input("Enter the number of your choice: ")
    plot_type = None
    sma_period = None

    if plot_choice == '1':
        plot_type = 'sma'
        sma_period = int(input("Enter SMA period (0 to skip): "))
    elif plot_choice == '2':
        plot_type = 'bollinger'
        sma_period = int(input("Enter Bollinger Bands period (0 to skip): "))
    elif plot_choice == '3':
        plot_type = 'rsi'
    elif plot_choice == '4':
        plot_type = 'macd'
    elif plot_choice == '5':
        plot_type = 'ichimoku'
    else:
        print("Invalid choice. Exiting.")
        return

    # Download stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)  # Approximate number of days for given years
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    if stock_data.empty:
        print(f"No data found for ticker: {ticker}")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(stock_data.index, stock_data['Adj Close'], label=f'{ticker} Price', color='orange')

    # Plot based on the selected plot type
    if plot_type == 'sma' and sma_period:
        stock_data['SMA'] = stock_data['Adj Close'].rolling(window=sma_period).mean()
        plt.plot(stock_data.index, stock_data['SMA'], label=f'SMA {sma_period}', color='blue')

    elif plot_type == 'bollinger' and sma_period:
        sma = stock_data['Adj Close'].rolling(window=sma_period).mean()
        rstd = stock_data['Adj Close'].rolling(window=sma_period).std()
        stock_data['Bollinger High'] = sma + 2 * rstd
        stock_data['Bollinger Low'] = sma - 2 * rstd
        plt.plot(stock_data.index, stock_data['Bollinger High'], label='Bollinger High', color='green')
        plt.plot(stock_data.index, stock_data['Bollinger Low'], label='Bollinger Low', color='red')

    elif plot_type == 'rsi':
        delta = stock_data['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        plt.plot(stock_data.index, rsi, label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red')
        plt.axhline(30, linestyle='--', color='green')

    elif plot_type == 'macd':
        short_ema = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
        long_ema = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        plt.plot(stock_data.index, macd, label='MACD', color='blue')
        plt.plot(stock_data.index, signal, label='Signal Line', color='red')

    elif plot_type == 'ichimoku':
        high_9 = stock_data['High'].rolling(window=9).max()
        low_9 = stock_data['Low'].rolling(window=9).min()
        stock_data['Tenkan-sen'] = (high_9 + low_9) / 2
        
        high_26 = stock_data['High'].rolling(window=26).max()
        low_26 = stock_data['Low'].rolling(window=26).min()
        stock_data['Kijun-sen'] = (high_26 + low_26) / 2
        
        stock_data['Senkou Span A'] = ((stock_data['Tenkan-sen'] + stock_data['Kijun-sen']) / 2).shift(26)
        stock_data['Senkou Span B'] = ((stock_data['High'].rolling(window=52).max() + stock_data['Low'].rolling(window=52).min()) / 2).shift(26)
        stock_data['Chikou Span'] = stock_data['Adj Close'].shift(-26)
        
        plt.plot(stock_data.index, stock_data['Tenkan-sen'], label='Tenkan-sen', color='cyan')
        plt.plot(stock_data.index, stock_data['Kijun-sen'], label='Kijun-sen', color='magenta')
        plt.plot(stock_data.index, stock_data['Senkou Span A'], label='Senkou Span A', color='green')
        plt.plot(stock_data.index, stock_data['Senkou Span B'], label='Senkou Span B', color='red')
        plt.plot(stock_data.index, stock_data['Chikou Span'], label='Chikou Span', color='blue')

    plt.title(f'{ticker} Stock Price Over the Last {years} Years', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Value', color='white')
    plt.legend()

    plt.gca().set_facecolor('black')  # Background color of the plot
    plt.gca().tick_params(axis='both', colors='grey')  # Color of the ticks
    plt.grid(color='grey', linestyle='--', linewidth=0.5)  # Grid color and style

    plt.gcf().patch.set_facecolor('black')  # Background color of the figure
    plt.show(block=False)




def plot_portfolio_performance_chart(years):
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    end_date = datetime.now()
    # Calculate start_date with float year input (handles partial years)
    start_date = end_date - timedelta(days=float(years * 365))  # Use int() to handle fractional days
    
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
    ticker_symbols_input = input("Enter ticker symbols separated by commas (e.g., AAPL,MSFT,GOOGL) or press Enter to skip: ")
    
    if ticker_symbols_input.strip():  # Check if input is not empty
        ticker_symbols = [symbol.strip() for symbol in ticker_symbols_input.split(',')]
        
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
                numbers_input = input("Enter the numbers of the titles you want to search, separated by commas (e.g., 1,3,5) or press Enter to skip: ")
                
                if numbers_input.strip():  # Check if input is not empty
                    numbers = [int(num.strip()) for num in numbers_input.split(',')]

                    # Validate and search links
                    if all(1 <= number <= len(all_headlines) for number in numbers):
                        links = [all_headlines[number - 1][2] for number in numbers]
                        search_links_on_google(links)
                        break
                    else:
                        print(f"Invalid numbers. Please enter numbers between 1 and {len(all_headlines)}.")
                else:
                    print("No titles selected for search. Exiting.")
                    break
            except ValueError:
                print("Invalid input. Please enter valid numbers separated by commas.")
    else:
        print("No ticker symbols entered. Exiting news function.")

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

def compare():
    tickers = input("Enter stock tickers (separated by commas): ").split(',')

    for ticker in [t.strip() for t in tickers]:
        stock = yf.Ticker(ticker)
        stock_info = stock.history(period="1d")
        stock_summary = stock.info
        
        # Fetching quarterly financials and cash flow
        quarterly_financials = stock.quarterly_financials
        quarterly_cashflow = stock.quarterly_cashflow
        
        # Calculate TTM Gross Profit
        if not quarterly_financials.empty and 'Gross Profit' in quarterly_financials.index:
            quarterly_financials = quarterly_financials.T.sort_index(ascending=False)
            gross_profit_quarters = quarterly_financials.loc[:, 'Gross Profit'].dropna().head(4)
            ttm_gross_profit = gross_profit_quarters.sum() if len(gross_profit_quarters) == 4 else "Not enough data"
            ttm_gross_profit = f"{round(ttm_gross_profit):,}" if isinstance(ttm_gross_profit, (int, float)) else ttm_gross_profit
        else:
            ttm_gross_profit = "No data available"
        
        # Calculate TTM Free Cash Flow
        if not quarterly_cashflow.empty and 'Free Cash Flow' in quarterly_cashflow.index:
            quarterly_cashflow = quarterly_cashflow.T.sort_index(ascending=False)
            free_cash_flow_quarters = quarterly_cashflow.loc[:, 'Free Cash Flow'].dropna().head(4)
            ttm_free_cash_flow = free_cash_flow_quarters.sum() if len(free_cash_flow_quarters) == 4 else "Not enough data"
            ttm_free_cash_flow_numeric = ttm_free_cash_flow if isinstance(ttm_free_cash_flow, (int, float)) else None
            ttm_free_cash_flow = f"{round(ttm_free_cash_flow):,}" if isinstance(ttm_free_cash_flow, (int, float)) else ttm_free_cash_flow
        else:
            ttm_free_cash_flow = "No data available"
            ttm_free_cash_flow_numeric = None
        
        # Fetch other data
        book_value = stock_summary.get('bookValue', None)
        latest_price = round(stock_info['Close'].iloc[-1]) if not stock_info.empty else "No data available"
        market_cap = stock_summary.get('marketCap', None)
        shares_outstanding = stock_summary.get('sharesOutstanding', None)
        
        if market_cap is not None:
            market_cap = f"{round(market_cap):,}"
        else:
            market_cap = "No data available"
        
        if shares_outstanding is not None:
            shares_outstanding = float(shares_outstanding)
            shares_outstanding_display = f"{round(shares_outstanding):,}"
        else:
            shares_outstanding = None
            shares_outstanding_display = "No data available"
        
        # Calculate FCF per Share
        if ttm_free_cash_flow_numeric is not None and shares_outstanding is not None:
            fcf_per_share = round(ttm_free_cash_flow_numeric / shares_outstanding, 2)
            fcf_per_share = f"${fcf_per_share:,.2f}"
        else:
            fcf_per_share = "No data available"
        
        if book_value is not None:
            book_value = f"{round(book_value):,}"
        else:
            book_value = "No data available"
        
        print(f"Ticker: {ticker.upper()}")
        print(f"Price: ${latest_price}")
        print(f"MC: ${market_cap}")
        print(f"TTM Profit: ${ttm_gross_profit}")
        print(f"TTM Free Cash Flow: ${ttm_free_cash_flow}")
        print(f"Shares Issued: {shares_outstanding_display}")
        print(f"BV/SH: ${book_value}")
        print(f"FCF/SH: {fcf_per_share}\n")

    #if __name__ == "__main__":
        #compare()


def main():
    while True:
        print("\nMenu:")
        print("[ch]   [news] [cc]")
        print("[gm]   [des]  [fvn]")
        print("[sa]   [fv]   [hol] ")
        print("[ins]  [sum]  [fs] ")
        print("[rs]   [port] [add]")
        print("[edit] [10k]  [10q] ")
        print("[op]   [sim]  [ovs]")
        print("[vic]  [gain] [compare]")
        print("[q]")
        

        choice = input("Choose an option: ").strip()
        
        if choice == 'ch':
            plot_stock_price()
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
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}/financials"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'sum':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'rs':
            ticker = input("Enter ticker: ").strip().upper()
            url = f"https://www.roic.ai/quote/{ticker}/ratios"
            import webbrowser
            webbrowser.open(url)
        elif choice == 'port':
            try:
                years = int(input("Enter number of years for portfolio performance chart: ").strip())
                if years <= 0:
                    raise ValueError("The number of years must be a positive integer.")
            except ValueError as e:
                print(f"Invalid input: {e}")
            else:
                plot_portfolio_performance_chart(years)
        elif choice == 'add':
            add_position()
        elif choice == 'edit':
            remove_position()
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
        elif choice == 'compare':
            compare()
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
