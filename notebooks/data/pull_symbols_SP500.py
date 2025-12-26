import pandas as pd

url = "https://www.slickcharts.com/sp500"

# Slickcharts table has the S&P 500 components
tables = pd.read_html(url)
df = tables[0]  # first table on the page

# Keep just the symbol column
symbols = df["Symbol"].astype(str).str.upper()

# Save as data/sp500_symbols.csv with a 'symbol' column
out_path = "data/sp500_symbols.csv"
symbols.to_frame(name="symbol").to_csv(out_path, index=False)

print(f"Saved {len(symbols)} symbols to {out_path}")
print(symbols.head())
