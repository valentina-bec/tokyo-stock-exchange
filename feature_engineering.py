import pandas as pd
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from datetime import *
import numpy as np


# fill nan values 
def fill_and_drop_na_values(df):
    """forward fill for each security code

    Args:
        df (): price_strock.csv
    """
    # fill nan values for expected dividends with 0
    df.ExpectedDividend.fillna(0)

    # create empty dataframe with columns
    stocks = pd.DataFrame(columns=df.columns)

    # doing for loop for every security
    for i in tqdm(df.SecuritiesCode.unique()):
    
        # creating query from dataframe with all rows for one stock
        query = df.query('SecuritiesCode == @i')

        # applying forward fill on query
        query = query.ffill()

        # updating dataframe, adding filled query
        stocks = pd.concat([stocks, query], axis=0)
    
    # drop remaining na values
    stocks.dropna(axis=0)

    # convert datetime 

    stocks['Date'] = pd.to_datetime(stocks['Date']) 



    # init file to convert to csv
    #file = 'data/train_files/stock_prices_wo_na.csv'

    # converting final dataframe to csv file
    #stocks.to_csv(file)

    return stocks

# Adjusting price: 
def adjust_price(DataFrame):
    """adjusting price with AdjustmentFactor 
     for stock_price Dataframe

    Args:
        DataFrame (pd.DataFrame): Stock Price
    Returns:
        DataFrame: _description_
    """

    # adjusted price for each code
    def adjust_price_slide(df):
        # cumulative adjustment factor considering the day shift
        df.loc[:,'CAF'] = df['AdjustmentFactor'].cumprod().shift(1)
        # fill nan values
        df.CAF.fillna(1, inplace=True)
        # prices to be adjusted
        prices =[ 'Open', 'High', 'Low', 'Close']

        for x in prices:
            #df['ad_' + str(x)] = df[x] / df['CAF']
            df.loc[:,'ad_' + str(x)]  = df[x] / df['CAF']
        
        # adjust volume
        df['ad_Volume'] = df['Volume'] * df['CAF']
        df.drop('CAF', axis=1, inplace=True)

        # adjust target
        df.sort_values('Date')
        df['ad_Close_1'] = df['ad_Close'].shift(-1)
        df['ad_Close_2'] = df['ad_Close'].shift(-2)
        df['ad_Target'] = (df['ad_Close_2']-df['ad_Close_1'])/df['ad_Close_1']
        df.drop(['ad_Close_1', 'ad_Close_2'], axis=1, inplace=True)

        return df

    codes = set(DataFrame.SecuritiesCode.unique())

    # create an empty new DataFrame
    adjusted_data = pd.DataFrame(columns = DataFrame.columns)

    for i in tqdm(codes):
        df = DataFrame.query('SecuritiesCode ==@i')
        adjusted_df = adjust_price_slide(df)
        adjusted_data = pd.concat([adjusted_data, adjusted_df ], axis=0)
    
    # convert data again to datetime
    adjusted_data['Date'] = pd.to_datetime(adjusted_data['Date']) 


    return adjusted_data

# imputing finances
def fill_finances_knn(financial, prices):

    # 
    financial['Day'] = financial['Date'].apply(lambda x: datetime.strftime(x,'%d')).apply(pd.to_numeric)
    financial['Month'] = financial['Date'].apply(lambda x: x.month).apply(pd.to_numeric)
    financial['Year'] = financial['Date'].apply(lambda x: x.year).apply(pd.to_numeric)

    # Object data to numeric data
    liste =[ 'Day', 'Month', 'Year', 'SecuritiesCode', 'Profit', 'NetSales', 'OperatingProfit', 'BookValuePerShare',
        'ForecastDividendPerShareFiscalYearEnd',
        'ForecastDividendPerShareAnnual', 'ForecastNetSales',
        'ForecastOperatingProfit', 'ForecastOrdinaryProfit', 'ForecastProfit',
        'ForecastEarningsPerShare']

    # drop ForecastRevision
    financial = financial.query('TypeOfDocument != ["ForecastRevision", "ForecastRevision_REIT"]')

    financial = financial[liste] 
    financial = financial.replace('－', np.nan, regex = True )  
    financial_num = financial.apply(lambda x: pd.to_numeric(x, errors = 'ignore'))

    # Selecting only matching Codes from financial data
    sec_codes = [x for x in prices.SecuritiesCode.unique() if x in financial.SecuritiesCode.unique()]
    df_pred = pd.DataFrame()

    # Filling missing values with KNN
    for i in tqdm(sec_codes):

        imputer = KNNImputer(n_neighbors=1, weights='distance', metric='nan_euclidean')
        df_current = pd.DataFrame(imputer.fit_transform(financial_num.query('SecuritiesCode == @i')))
        df_pred = pd.concat([df_pred, df_current])

    # Reassigning column names
    df_pred.columns = liste

    # Recreating Date for merge
    df_pred["Date"] = pd.to_datetime(dict(year=df_pred.Year, month=df_pred.Month, day=df_pred.Day))

    # Placing Date as first column
    col = df_pred.pop("Date")
    df_pred.insert(0, col.name, col)

    # Selecting only Columns without NaN
    pred_final = ['Date', 'Day', 'Month', 'Year', 'SecuritiesCode', 'Profit', 'NetSales']
    df_pred = df_pred[pred_final]

    return df_pred