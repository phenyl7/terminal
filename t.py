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





def portfolio_check():
    data = []
    total_value = 0
    fig, ax = plt.subplots(figsize=(12, len(portfolio) * 0.6 + 2))
    ax.axis('off')

    table_data = []
    for ticker, details in portfolio.items():
        stock_data = get_stock_data(ticker)
        last_close_price = stock_data['price_current']
        
        if last_close_price is not None:
            dollar_change = (last_close_price - details['cost_basis']) * details['shares']
            percent_change = ((last_close_price - details['cost_basis']) / details['cost_basis']) * 100
            position_value = details['shares'] * last_close_price
            total_value += position_value

            table_data.append([
                ticker, 
                round(details['shares'], 2),  # Round shares to 2 decimal places
                round(dollar_change, 2), 
                round(percent_change, 2), 
                round(position_value, 2)
            ])
    
    df = pd.DataFrame(table_data, columns=['Ticker', 'Shares', '$ ∆', '% ∆', 'PV'])
    
    total_row = pd.DataFrame([['tpv', '', '', '', f'${total_value:,.2f}']], columns=df.columns)
    df = pd.concat([df, total_row], ignore_index=True)
    
    print(df)
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    for (i, j), cell in table._cells.items():
        cell.set_text_props(color='orange')  # Neon green text
        cell.set_facecolor('black')  # Black background
        cell.set_edgecolor('gray')  # Gray grid lines
    
    plt.title("Current Portfolio", fontsize=10, color='orange')
    plt.gcf().set_facecolor('black')
    plt.show(block=False)

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

def main():
    while True:
        print("\nMenu:")
        print("[1] Chart             [2] News")
        print("[3] Commodities       [4] Crypto")
        print("[5] SA                [6] Finviz")
        print("[7] 10-K              [8] 10-Q")
        print("[9] StockCharts       [10] Insiders")
        print("[11] Quote            [12] Financials")
        print("[13] Ratios           [14] Portfolio Performance Chart")
        print("[15] New Entry        [16] Edit Port")
        print("[17] Portfolio Check  [q] Quit")

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
            webbrowser.open("https://finviz.com/news.ashx")
        elif choice == '3':
            webbrowser.open("https://tradingeconomics.com/commodities")
        elif choice == '4':
            webbrowser.open("https://tradingeconomics.com/crypto")
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
            portfolio_check()
        elif choice == 'q':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
