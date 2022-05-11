import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# missing values
def missingValues(dataframe):
	missing = {}
	# for a singel column
	if isinstance(dataframe, type(pd.Series({'a': 1, 'b': 2, 'c': 3}))):
		n_miss = dataframe.isnull().sum()
		perc = n_miss / dataframe.shape[0] * 100
		print(' missing Values: %d (%.1f%%)' % (n_miss, perc))
	
	else: 
		cols = dataframe.columns


		for i in range(len(cols)):
			# count number of rows with missing values
			n_miss = dataframe[dataframe.columns[i]].isnull().sum()
			perc = n_miss / dataframe.shape[0] * 100
			missing[dataframe.columns[i]] = (n_miss, round(perc,2))
			#print(dataframe.columns[i] + ' missing Values: %d (%.1f%%)' % (n_miss, perc))
		
		missing_df = pd.DataFrame(missing).T
		missing_df.columns = ['N_missing', 'Percentage' ]
		display(missing_df.sort_values('Percentage', ascending=False))
		


# plot one stock
def plot_stock(df, Code, feature='Target', color='blue'):
    df = df.query('SecuritiesCode==@Code')
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df, y=feature, x='Date', label=Code, color=color)
    

#create a dataframe for only one security code:
def df_security_code(df, code=7203): # Toyotagit a
	return df.query('SecuritiesCode == @code')


def print_shape(df, missing=True):
	def human_format(num):
		num = float('{:.3g}'.format(num))
		magnitude = 0
		while abs(num) >= 1000:
			magnitude += 1
			num /= 1000.0
		return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

	print (f' Shape:\n {"----"*10}')
	print(f' Observations:   {human_format(df.shape[0])}')
	print(f' Features:       {df.shape[1]}')
	print(f' Feature Date:    {df["Date"].dtype}' )
	print(f'{"----"*10}')
	print(f'{df.columns}')
	print(f'{"----"*10}')
	if missing: missingValues(df)

def plot_corr(df):
	fig=plt.figure(figsize=(14,7))

	matrix = np.triu(df.corr())

	sns.heatmap(df.corr(), cmap='YlGnBu', annot=True, linewidth=0.6, mask=matrix)

	plt.title('Correlation table', fontsize=18)	


def date_range(df, date='Date'):
    date = df.date
    date_desc = date.describe(datetime_is_numeric=True)
    date_min = date_desc.loc['min']
    date_max = date_desc.loc['max']
    print('Data from {:%Y-%m-%d} to {:%Y-%m-%d}'
            .format(date_min, date_max))
    print('Data observations {} '.format(
                    date.nunique()))


def plot_stock(df, code, feature='Close'):
    df = df.query('SecuritiesCode==@code')
    plt.figure(figsize=(20,5))
    sns.lineplot(data=df, y=feature, x='Date', label=code)
    plt.title(f'{code} {feature}')
