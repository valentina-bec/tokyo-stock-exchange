import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# missing values
def missingValues(dataframe):
	for i in range(len(dataframe.columns)):
		# count number of rows with missing values
		n_miss = dataframe[dataframe.columns[i]].isnull().sum()
		perc = n_miss / dataframe.shape[0] * 100
		print(dataframe.columns[i] + ' missing Values: %d (%.1f%%)' % (n_miss, perc))
    


# plot one stock
def plot_stock(df, Code, feature='Target', color='blue'):
    df = df.query('SecuritiesCode==@Code')
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df, y=feature, x='Date', label=Code, color=color)
    

#create a dataframe for only one security code:
def df_security_code(df, code):
	return df.query('SecuritiesCode == @code')