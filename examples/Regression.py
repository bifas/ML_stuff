import pandas as pd
import quandl

quandl.ApiConfig.api_key = "ex_ZuzsqWF78WrwCPe4X"

if __name__ == "__main__":
    df = quandl.get('WIKI/GOOGL')

    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

    print(df.head())
