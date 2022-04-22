#from msilib.schema import Feature
import pandas as pd
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import KNNImputer
from datetime import *
import numpy as np
from sklearn.preprocessing import LabelEncoder



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
        df = DataFrame.query('SecuritiesCode ==@i').sort_values('Date')
        adjusted_df = adjust_price_slide(df)
        adjusted_data = pd.concat([adjusted_data, adjusted_df ], axis=0)
    
    # convert data again to datetime
    adjusted_data['Date'] = pd.to_datetime(adjusted_data['Date']) 


    return adjusted_data

# create new features for stock prices
def price_new_features(df):

    # features lag 1
    def features_lag(df_code, feat, lag=1):
        name = feat + "_lag" + str(lag)
        return name, df_code[str(feat)].shift(lag)
    
    def RSI(df_serie, periods=14, ema=True):
        """Relative Strength Index"""
        close_delta = df_serie.diff() # .dropna()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema == True: # exponential moving average
            ma_up = up.ewm(com= periods - 1, adjust=True, min_periods = periods).mean()
            ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        
        else: # using moving average
            ma_up = up.rolling(window = periods, adjust=False).mean()
            ma_down = down.ewm(window = periods, adjust=False).mean()
        
        rs = ma_up / ma_down
  
        return 100 - (100 / (1+rs))


    def return_stock(df_serie, type='D'):
        if type == 'M':
            return df_serie.resample('M').ffill().pct_change()
        
        if type == 'D':
            return df_serie.pct_change()

        if type == 'cum_M':
            return (df_serie.resample('M').ffill().pct_change() + 1).cumprod()


    def SMA(df_code, feat, period = 10): # period 5
        """ Simple moving average"""
        name = name = feat + "_sma" + str(period)
        sma = df_code[feat].rolling(window=period).mean()
        return name, sma

    def MACD(df_code):
        ema26 = df['ad_Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        ema12 = df['ad_Close'].ewm(span=12, adjust=False, min_periods=12).mean()

        macd = ema12 - ema26

        # Get the 9-Day EMA of the MACD for the Trigger line
        macd_ema9 = macd.ewm(span=9, adjust=False, min_periods=9).mean()

        # calculate the difference
        macd_diff = macd - macd_ema9

        df['macd'] = df.index.map(macd)
        df['macd_h'] = df.index.map(macd_diff)
        df['macd_s'] = df.index.map(macd_ema9)

        return df['macd'],  df['macd_h'], df['macd_s']



    stocks = pd.DataFrame(columns=df.columns)

    codes = df.SecuritiesCode.unique()
    # doing for loop for every security
    for i in tqdm(codes):
        df_code = df.query('SecuritiesCode ==@i').sort_values('Date')
        
        # features
        features = ['ad_Close', 'ad_Open', 'ad_High' , 'ad_Low', 'ad_Volume']
        for feat in features:
            # lag 1
            name_l, lag_df =  features_lag(df_code, feat)
            df[name_l] = lag_df

            # simple moving average
            name_sma , sma_df = SMA(df_code, feat)
            df[name_sma] = sma_df

        # RSI: Relative Strengt index
        df_code['RSI'] = RSI(df_code['ad_Close'])

        # Return / default daily, options montly cummulativ
        df_code['Return'] = return_stock(df_code['ad_Close'])

        # MACD: Moving Average Convergence Divergence

        df_code['MACD'] , df_code['MACD_h'], df_code['MACD_s'] = MAC(df_code)

# encode flag in stock prices
def encode_flag(df, feature = "SupervisionFlag"):
    """encode prices["SupervisionFlag"]

    Args:
        df_series (_type_): _description_

    Returns:
        _type_: _description_
    """
    enc = LabelEncoder()
    enc.fit(df[feature])
    return enc.transform(df[feature])


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