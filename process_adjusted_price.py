import pandas as pd
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# Function importing Dataset
def importdata(): 
    if len(sys.argv) != 2:
        sys.exit("Usage: python process_adjusted_price.py file.csv \n file: data/train_files/stock_prices.csv")
    # data/train_files/stock_prices.csv

    data = pd.read_csv(sys.argv[1], sep= ',', header = 0, parse_dates=[1]) 
      
    # Printing the dataset shape 
    print ("Dataset Length: {:,}".format( len(data))) 
    print ("Dataset Shape: ", data.shape) 
      
    # Printing the dataset observations 
    print ("Dataset: \n",data.head(2)) 
    return data 


# Adjusting price: 
def adjust_price(df):
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

    return df


if __name__ == "__main__":
    # import data
    data = importdata()

    # create an empty new DataFrame
    adjusted_data = pd.DataFrame(columns = data.columns)

    #selected_codes = [8876, 6630, 7453, 7638]
    # iterate for each security code
    for i in tqdm(data.SecuritiesCode.unique()):
        df = data.query('SecuritiesCode ==@i')
        adjusted_df = adjust_price(df)
        adjusted_data = pd.concat([adjusted_data, adjusted_df ], axis=0)
    
    del data
    file = 'data/train_files/stock_prices_ad.csv'
    print(f'----'*15)
    print(f'Data: {adjusted_data.shape}, saved in: {file}')
    print(f'data columns: {adjusted_data.columns}')
    adjusted_data.to_csv(file)

