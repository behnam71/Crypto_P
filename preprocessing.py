import pandas as pd


def load_dataset(*, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, low_memory=False, index_col=[0])
    """
    df = pd.read_csv(file_name, skiprows=1)
    df.drop(columns=['unix', 'symbol', 'Volume BTC', 'tradecount'], inplace=True)
    df = df.rename({Volume USDT: "volume"}, axis=1)

    temp = ''
    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    for i in range(0, len(df)):
        if df.loc[i, 'date'][-2:]!='AM' and df.loc[i, 'date'][-2:]!='PM':
            if df.loc[i, 'date'] != temp:
                d = datetime.strptime(df.loc[i, 'date'], "%Y-%m-%d %H:%M:%S")
                df.loc[i, 'date'] = d.strftime("%Y-%m-%d %I-%M-%S %p")
                temp = df.loc[i, 'date']
        else:
            df.loc[i, 'date'] = df.loc[i, 'date'][:13] + '-00-00 ' + df.loc[i, 'date'][-2:]

    # Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])
    
    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')
    df.to_csv(file_name)
    """
    return df
