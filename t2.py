import requests
import webbrowser
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

def open_sec_link(ticker, form_type):
    if form_type == "10k":
        url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-K"
    elif form_type == "10q":
        url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}&filter_forms=10-Q"
    elif form_type == "cf":
        url = f"https://www.sec.gov/edgar/search/?r=el#/dateRange=all&entityName={ticker}"
    elif form_type == "sta":
        url = f'https://stockanalysis.com/stocks/{ticker}'
    elif form_type == "g":
        url = f'https://stockcharts.com/sc3/ui/?s={ticker}'
    elif form_type == "q": 
        url = f'https://finance.yahoo.com/quote/{ticker}/'
    elif form_type == "n": 
        url = f'https://stockanalysis.com/stocks/{ticker}'
    elif form_type == "fa": 
        url = f'https://stockanalysis.com/stocks/{ticker}/financials/'
    elif form_type == 'sa':
        url = f'https://seekingalpha.com/symbol/{ticker}'
    elif form_type == 'fv':
        url = f'https://finviz.com/quote.ashx?t={ticker}&p=d'
    elif form_type == 'hds':
        url = f'https://whalewisdom.com/stock/{ticker}'
    elif form_type == 'ins':
        url = f'http://openinsider.com/search?q={ticker}'
    else:
        print("Invalid form type.")
        return

    webbrowser.open(url)

def get_cik_for_ticker(ticker):
    """Fetches the CIK for a given ticker."""
    base_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Your Name (your_email@example.com)"
    }

    response = requests.get(base_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for company in data.values():
            if company['ticker'].lower() == ticker.lower():
                return company['cik_str']
    else:
        print(f"Failed to retrieve CIK data. Status code: {response.status_code}")
        return None

def get_edgar_filings(cik, filing_type=None, count=10):
    """Fetches recent filings for a company with a given CIK."""
    base_url = "https://data.sec.gov/submissions/"
    headers = {
        "User-Agent": "Your Name (your_email@example.com)"
    }

    url = f"{base_url}CIK{cik:0>10}.json"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        filings = data.get("filings", {}).get("recent", {})
        if not filings:
            print("No filings found.")
            return []

        indices = range(len(filings.get("form", [])))
        if filing_type:
            indices = [i for i in indices if filings['form'][i] == filing_type]
        indices = indices[:count]

        filing_links = []
        for i in indices:
            accession_number = filings['accessionNumber'][i].replace('-', '')
            primary_document = filings['primaryDocument'][i]
            filing_link = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
            
            print(f"{filings['form'][i]}")
            print(f"{filings['filingDate'][i]}")
            print(f"{filings['accessionNumber'][i]}")
            print(f"{filings['primaryDocument'][i]}")
            #print(f"Filing Link: {filing_link}")
            print("-" * 3)
            
            filing_links.append(filing_link)

        # Open all filing links in a web browser
        for link in filing_links:
            webbrowser.open(link)

        return filing_links

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return []
    

def calculate_rsi(data, window=14):
    """Calculate the Relative Strength Index (RSI)."""
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def plot_stock_chart(ticker, period):
    """Fetches and plots the stock price for the given ticker and period."""
    # Define period mapping for yfinance
    period_mapping = {
        "1d": {"period": "1d", "interval": "5m"},
        "5d": {"period": "5d", "interval": "15m"},
        "3m": {"period": "3mo", "interval": "1d"},
        "6m": {"period": "6mo", "interval": "1d"},
        "ytd": {"period": "ytd", "interval": "1d"},
        "1y": {"period": "1y", "interval": "1d"},
        "5y": {"period": "5y", "interval": "1wk"},
    }

    if period not in period_mapping:
        print("Invalid period. Valid periods: 1d, 5d, 3m, 6m, ytd, 1y, 5y.")
        return

    try:
        # Fetch the stock data using yfinance
        data = yf.download(ticker, period=period_mapping[period]["period"], interval=period_mapping[period]["interval"])

        if data.empty:
            print(f"No data found for {ticker} with period '{period}'.")
            return

        # Convert data to appropriate format for mplfinance
        data.index = pd.to_datetime(data.index)
        mpf_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Calculate the RSI
        data['RSI'] = calculate_rsi(data)

        market_colors = mpf.make_marketcolors(
            up='cyan',    # Custom color for up candles
            down='red',    # Custom color for down candles
            wick='inherit',  # Custom color for wicks
            edge='white',
            volume = '#000000'
        )

        # Define a custom style to adjust font sizes and set plot area background color
        custom_style = mpf.make_mpf_style(
            base_mpf_style='default',
            marketcolors=market_colors,
            rc={'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.color': '#c9c9c9', 'axes.labelcolor': '#c9c9c9', 'xtick.color': '#c9c9c9', 'ytick.color': '#c9c9c9', 'grid.color': '#2e2e2e'},
            facecolor='#000000',  # Light gray background for the plot area
        )

        # Create the RSI subplot
        add_rsi = [
            mpf.make_addplot(data['RSI'], panel=2, color='#c9c9c9', ylabel='RSI'),
            mpf.make_addplot([70] * len(data), panel=2, color='red', linestyle='--'),  # Red line at 70
            mpf.make_addplot([30] * len(data), panel=2, color='green', linestyle='--'),  # Green line at 30
            mpf.make_addplot(data['RSI'].where(data['RSI'] >= 70), panel=2, color='cyan', alpha=1),  # Area above 70
            mpf.make_addplot(data['RSI'].where(data['RSI'] <= 30), panel=2, color='cyan', alpha=1)  # Area below 30
        ]

        # Plot the data with reduced text size and return the figure
        fig, axes = mpf.plot(
            mpf_data,
            type='candle',
            style=custom_style,
            volume=True,
            tight_layout=True,
            scale_padding=dict(left=0.7, top=1, right=1.5),
            figsize=(10, 8),  # Adjust the figure size as needed
            returnfig=True,  # This is important to get the figure object
            addplot=add_rsi
        )
        fig.suptitle(f'{ticker} {period}', fontsize=8, color='#c9c9c9')
        # Set the figure background color (change '#E6E6FA' to any color you prefer)
        fig.patch.set_facecolor('#000000')  # Light lavender color for the margins

        # Display the plot
        plt.show(block=False)

    except Exception as e:
        print(f"Failed to download: {ticker}. Error: {e}")




def print_commands():
    """Prints a list of available commands."""
    commands = {
        "ticker cf": "opens link for all filings (e.g., 'AAPL CF').",
        "ticker 10-k #": "opens links for 10k filings (e.g., 'AAPL 10-K 5').",
        "ticker 10-q #": "opens links for 10q filings (e.g., 'AAPL 10-Q 5').",
        "ticker sta": "Open Stock Analysis page for the specified ticker.",
        "ticker g": "Open StockCharts for the specified ticker.",
        "ticker q": "Open Yahoo Finance for the specified ticker.",
        "ticker n": "Open Stock Analysis page for the specified ticker.",
        "ticker fa": "Open Financials page for the specified ticker.",
        "ticker sa": "open seeking alpha link for stock",
        "ticker fv": "open finviz link for stock",
        "ticker hds": "open openinsider link for stock",
        "ticker ins": "open openinsider link for stock",
        "ticker g 1d/5d/1mo/3m/6m/ytd/1y/5y": "Plot a candlestick chart for the specified period",
        "?": "Display this help message."
    }
    print("Available commands:")
    for command, description in commands.items():
        print(f"{command}: {description}")

def main():
    while True:
        user_input = input("> ").strip().split()
        if user_input[0].lower() == 'q':
            break
        elif user_input[0] == '?':
            print_commands()
            continue
        elif len(user_input) < 1:
            print("No input provided.")
            continue
        
        ticker = user_input[0].upper()

        # Check if the command is for SEC links or filings
        if len(user_input) == 2:
            form_type = user_input[1].lower()
            open_sec_link(ticker, form_type)
        elif len(user_input) == 3:
            if user_input[1].lower() == 'g' and user_input[2].lower() in ["1d", "5d", "1mo", "3m", "6m", "ytd", "1y", "5y"]:
                period = user_input[2].lower()
                plot_stock_chart(ticker, period)
            else:
                filing_type = user_input[1].upper()
                count = int(user_input[2]) if user_input[2].isdigit() else 10
                
                cik = get_cik_for_ticker(ticker)
                if cik:
                    print(f"CIK for {ticker}: {cik}")
                    get_edgar_filings(cik=cik, filing_type=filing_type, count=count)
                else:
                    print(f"CIK for ticker '{ticker}' not found.")

if __name__ == "__main__":
    main()
