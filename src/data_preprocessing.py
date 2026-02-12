import pandas as pd

def load_and_preprocess(filepath):

    # Load CSV
    data = pd.read_csv(filepath)

    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort by date
    data = data.sort_values('Date')

    # Aggregate sales by date
    data = data.groupby('Date')['Weekly_Sales'].sum()

    # Set weekly frequency explicitly
    data = data.asfreq('W-FRI')

    return data
