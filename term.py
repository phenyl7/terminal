import requests
import webbrowser

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
            print("-" * 40)
            
            filing_links.append(filing_link)

        # Open all filing links in a web browser
        for link in filing_links:
            webbrowser.open(link)

        return filing_links

    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return []

def print_commands():
    """Prints a list of available commands."""
    commands = {
        "TICKER CF": "Open SEC links for the specified ticker and form type (e.g., 'AAPL CF').",
        "TICKER 10-K COUNT": "Get the recent 10-K filings for the specified ticker (e.g., 'AAPL 10-K 5').",
        "TICKER 10-Q COUNT": "Get the recent 10-Q filings for the specified ticker (e.g., 'AAPL 10-Q 5').",
        "TICKER STA": "Open Stock Analysis page for the specified ticker.",
        "TICKER G": "Open StockCharts for the specified ticker.",
        "TICKER Q": "Open Yahoo Finance for the specified ticker.",
        "TICKER N": "Open Stock Analysis page for the specified ticker.",
        "TICKER FA": "Open Financials page for the specified ticker.",
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
        elif len(user_input) >= 3:
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
