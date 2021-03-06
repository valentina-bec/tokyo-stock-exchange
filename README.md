# JPX Tokyo Stock Exchange Prediction

Success in any financial market requires one to identify solid investments. When a stock or derivative is undervalued, it makes sense to buy. If it's overvalued, perhaps it's time to sell. While these finance decisions were historically made manually by professionals, technology has ushered in new opportunities for retail investors. Data scientists, specifically, may be interested to explore quantitative trading, where decisions are executed programmatically based on predictions from trained models.


## Data

The data and challenge were kindly privided by Kaggle: 

[jpx-tokyo-stock-exchange-prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/overview)

This dataset contains historic data for a variety of Japanese stocks and options. Your challenge is to predict the future returns of the stocks. 

### Exploratory data analysis 
- stock_list.csv
- train_files/financials.csv
- train_files/stock_prices.csv

## Data Preprocessing and feature engineering
### Data cleaning 
Data cleaning and filling is a process that is used to clean up and fill in missing data in data sets. This process can be used to improve the accuracy of data sets, as well as to improve the efficiency of analysis processes. Missing values were fill with forward fill method. 
- prices:  
    - Adjusted Price with AdjustmentFactor
    - remove seasonality
### New features
the original data includes essential indicators such as closing, open, high and low prices, as well as volume. With new features we introduce more sophisticated technical indicators:

Prices: 
- Prices lag: these are the prices with lag 1, i.e., from the previous day/period.

- RSI: relative strength index
- Log Return 
- SMA: simple moving average
- MACD: Moving Average Convergence/Divergence
- Volatility

for our predictions we include the financial data including as well new features:

- Profit growth
- Sales growth
- Margin
- Margin growth

files : <br>
[preprocessing.ipynb](preprocessing.ipynb)<br>
[feature_engineering.py](feature_engineering.py)


## Models

### VAR


### ARIMA


### ANN

- model training: [ann_model](modeling/ann_model.ipynb) <br>
- predictions:    [ann_predict](modeling/ann_predict.ipynb) <br>
- evaluation:     [ann_evaluate](modeling/ann_evaluate.ipynb) <br>


<br>
<br>


<!--

# ds-modeling-pipeline

Here you find a Skeleton project for building a simple model in a python script or notebook and log the results on MLFlow.

There are two ways to do it: 
* In Jupyter Notebooks:
    We train a simple model in the [jupyter notebook](notebooks/EDA-and-modeling.ipynb), where we select only some features and do minimal cleaning. The hyperparameters of feature engineering and modeling will be logged with MLflow

* With Python scripts:
    The [main script](modeling/train.py) will go through exactly the same process as the jupyter notebook and also log the hyperparameters with MLflow

Data used is the [coffee quality dataset](https://github.com/jldbc/coffee-quality-database).
-->
## Requirements:

- pyenv with Python: 3.9.4

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

The `requirements.txt` file contains the libraries needed for deployment.. of model or dashboard .. thus no jupyter or other libs used during development.

<!--

The MLFLOW URI should **not be stored on git**, you have two options, to save it locally in the `.mlflow_uri` file:

```BASH
echo http://127.0.0.1:5000/ > .mlflow_uri
```

This will create a local file where the uri is stored which will not be added on github (`.mlflow_uri` is in the `.gitignore` file). Alternatively you can export it as an environment variable with

```bash
export MLFLOW_URI=http://127.0.0.1:5000/
```

This links to your local mlflow, if you want to use a different one, then change the set uri.

The code in the [config.py](modeling/config.py) will try to read it locally and if the file doesn't exist will look in the env var.. IF that is not set the URI will be empty in your code.

## Usage

### Creating an MLFlow experiment

You can do it via the GUI or via [command line](https://www.mlflow.org/docs/latest/tracking.html#managing-experiments-and-runs-with-the-tracking-service-api) if you use the local mlflow:

```bash
mlflow experiments create --experiment-name 0-template-ds-modeling
```

Check your local mlflow

```bash
mlflow ui
```

and open the link [http://127.0.0.1:5000](http://127.0.0.1:5000)

This will throw an error if the experiment already exists. **Save the experiment name in the [config file](modeling/config.py).**

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python -m modeling.train
```

In order to test that predict works on a test set you created run:

```bash
python modeling/predict.py models/linear data/X_test.csv data/y_test.csv
```
-->
