def fill_and_drop_na_values(df):
    
    # fill nan values for expected dividends with 0
    df.ExpectedDividend.fillna(0)

    # create empty dataframe with columns
    stocks = pd.DataFrame(columns=df.columns)

    # doing for loop for every security
    for i in tqdm(df.SecuritiesCode.unique()):
    
        # creating query from dataframe with all rowws for one stock
        query = df.query('SecuritiesCode == @i')

        # applying forward fill on query
        query = query.ffill()

        # updating dataframe, adding filled query
        stocks = pd.concat([stocks, query], axis=0)
    
    # drop remaining na values
    stocks.dropna(axis=0)

    # init file to convert to csv
    file = 'data/train_files/stock_prices_wo_na.csv'

    # converting final dataframe to csv file
    stocks.to_csv(file)
