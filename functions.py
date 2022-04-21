import pandas as pd
import seaborn as sns


def missingValues(dataframe):

	for i in range(len(dataframe.columns)):
		# count number of rows with missing values
		n_miss = dataframe[dataframe.columns[i]].isnull().sum()
		perc = n_miss / dataframe.shape[0] * 100
		print(dataframe.columns[i] + ' missing Values: %d (%.1f%%)' % (n_miss, perc))
    pass

def plot_stock(df, Code, feature='Target'):
    df = df.query('SecuritiesCode==@Code')
    sns.lineplot(data=df, y=feature, x='Date', label=Code)
    pass