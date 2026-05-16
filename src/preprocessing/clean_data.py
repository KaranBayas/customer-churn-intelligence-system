def clean_data(df):
    df.drop(columns="customerId", inplace=True, errors="ignore")  # Remove 'Id' column if it exists 
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (example: fill with mean)
    df = df.fillna(df.mean())

    # Convert data types if necessary (example: convert 'date' column to datetime)
    

    return df
