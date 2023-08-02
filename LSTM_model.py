#!/usr/bin/env python
# coding: utf-8

# In[46]:


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from datetime import timedelta
import boto3
import os
import json
import math
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler



# In[47]:


with open('/Users/cedrix/Documents/aws.json', 'r') as f:
    credentials = json.load(f)

# Set environment variables
os.environ['AWS_ACCESS_KEY_ID'] = credentials ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = credentials ['AWS_SECRET_ACCESS_KEY']

# AWS S3 bucket
bucket = 'raw-stock-price'

# Load data from S3
  
def load_data_from_s3(file_name):
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    obj = s3.get_object(Bucket=bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    print("NaN values in data after loading:", df.isnull().sum().sum())
    return df

# Function to list all files in a specific S3 bucket folder
def list_files_in_s3_bucket(bucket_name, prefix):
    s3 = boto3.client('s3', aws_access_key_id=credentials['AWS_ACCESS_KEY_ID'], aws_secret_access_key=credentials['AWS_SECRET_ACCESS_KEY'])
    response = s3.list_objects(Bucket=bucket, Prefix=prefix)

    # Get a list of all the file names
    files = [item['Key'] for item in response['Contents']]

    # Extract the stock symbol from each file name
    stock_symbols = [file.split('/')[-1].split('_')[0] for file in files]

    return stock_symbols

def preprocess_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)  # Drop rows with missing data
    return df

def add_feature(df, feature, window):
    if feature == 'MA':
        close_col = df['adj_close']
        df['MA'] = close_col.rolling(window=window).mean()
    if feature == 'EMA':
        close_col = df['adj_close']
        df['EMA'] = close_col.ewm(span=window, adjust=False).mean()
    if feature == 'SO':
        high14 = df['high'].rolling(window).max()
        low14 = df['low'].rolling(window).min()
        df['%K'] = (df['close'] - low14) * 100 / (high14 - low14)
        df['%D'] = df['%K'].rolling(3).mean()

    # Drop rows with NaN values
    df.dropna(inplace=True)
        
    if df.isnull().values.any():
        print(f"NaN values introduced after adding {feature}")

    return df

def evaluate_model(model, X_test, y_test, metric):
    predictions = model.predict(X_test)
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(y_test, predictions))
    elif metric == 'mse':
        return mean_squared_error(y_test, predictions)
    elif metric == 'mape':
        return np.mean(np.abs((y_test - predictions) / y_test)) * 100
    else:
        print(f"Metric {metric} not recognized")
    return None



# In[66]:


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
        
        
    return np.array(Xs), np.array(ys)

def test_create_dataset():
    # Create a simple DataFrame
    df = pd.DataFrame({
        'A': range(10),
        'B': range(10, 20),
        'C': range(20, 30)
    })

    # Create X and y
    X = df[['A', 'B']]
    y = df['C']

    # Call create_dataset
    Xs, ys = create_dataset(X, y, time_steps=3)

    # Print the results
    print("Xs:")
    print(Xs)
    print("ys:")
    print(ys)

    # Call the test function
    test_create_dataset()
    
def reshape_data(data, time_steps=1):
    # Insert your choice of padding here. I'll use 0.
    padding = np.zeros((time_steps - 1, data.shape[1]))
    data = np.concatenate([padding, data])
    reshaped_data = np.array([data[i:i + time_steps] for i in range(data.shape[0] - time_steps + 1)])
    return reshaped_data


# In[67]:


def train_model(df, future_days, test_size, lstm_units, dropout, epochs, batch_size):


    try:
        # Apply shift operation
        df['Prediction'] = df['adj_close'].shift(-future_days)

        df_copy = df.copy()

        # Create X_predict using the shifted copy
        X_predict = np.array(df_copy.drop(['Prediction'], 1))[-future_days:]
        X_predict = np.array(df.drop(['Prediction'], 1))[-future_days:]

        X = df.drop(['Prediction'], axis=1)
        X = X[:-future_days]
        y = df['Prediction']
        y = y[:-future_days]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        print("NaN values in X_train:", np.isnan(X_train).sum())
        print("NaN values in X_test:", np.isnan(X_test).sum())
        print("NaN values in y_train:", np.isnan(y_train).sum())
        print("NaN values in y_test:", np.isnan(y_test).sum())
        
        print("Shape of X_train:", X_train.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of y_test:", y_test.shape)
        
        print("Shape of X:", X.shape)
        print("Shape of y:", y.shape)
        print("NaN values in X:", np.isnan(X).sum())
        print("NaN values in y:", np.isnan(y).sum())
        
        
        
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        time_steps = 1
        X_train, y_train = create_dataset(pd.DataFrame(X_train), pd.DataFrame(y_train), time_steps)
        X_test, y_test = create_dataset(pd.DataFrame(X_test), pd.DataFrame(y_test), time_steps)

        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout))
        model.add(LSTM(units=lstm_units))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

        with mlflow.start_run():
            mlflow.log_param("future_days", future_days)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("lstm_units", lstm_units)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[es],
                shuffle=False
            )

            # Log model
            mlflow.keras.log_model(model, "lstm")

            # Log metrics: RMSE, MSE and MAPE
            rmse = math.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            mse = mean_squared_error(y_test, model.predict(X_test))
            mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mape", mape)

        return model, X_train, X_test, y_train, y_test, X_predict, scaler

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        mlflow.end_run()
        
        
        
def run_model(file_name, ma_window=5, lstm_units=50, dropout=0.2, epochs=100, batch_size=32, test_size=0.2, future_days=30, rmse=True, mse=True, mape=True, display_at=0):
    try:
        # Load data from S3
        df = load_data_from_s3(file_name)
        print("NaN values in data after preprocessing:", df.isnull().sum().sum())



        # Preprocess data
        df = preprocess_data(df)
        print("NaN values in data after preprocessing:", df.isnull().sum().sum())

        # Define feature windows
        feature_windows = {
            'MA': ma_window,
        }

        # Add features to the data
        for feature in feature_windows:
            print(f"NaN values in data after adding {feature}:", df.isnull().sum().sum())
            df = add_feature(df, feature, feature_windows[feature])

        # Train model and evaluate
        model, X_train, X_test, y_train, y_test, X_predict, scaler = train_model(df, future_days, test_size, lstm_units, dropout, epochs, batch_size)
        evaluations = {}
        if rmse:
            evaluations['rmse'] = evaluate_model(model, X_test, y_test, 'rmse')
        if mse:
            evaluations['mse'] = evaluate_model(model, X_test, y_test, 'mse')
        if mape:
            evaluations['mape'] = evaluate_model(model, X_test, y_test, 'mape')

            
        # Print the dataframe before dropping the 'Prediction' column
        print(df)

        # Check if 'Prediction' column exists in the dataframe
        if 'Prediction' in df.columns:
            print("Prediction column exists in the dataframe.")
        else:
            print("Prediction column does not exist in the dataframe.")   
            
        
        # Generate prediction
        lstm_model_real_prediction = model.predict(np.array(df.drop(['Prediction'], 1)))
        
        data = data.reshape((data.shape[0], 1, data.shape[1]))

        lstm_model_real_prediction = model.predict(data)
        
        
        # Plot
        plot_results(df, lstm_model_real_prediction, lstm_model_predict_prediction, display_at, future_days)

        return model, evaluations, df
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None  # return None for each expected return value
    finally:
        mlflow.end_run()
        
        
def plot_results(df, lstm_model_real_prediction, lstm_model_predict_prediction, display_at, future_days):
    predicted_dates = [df.index[-1] + timedelta(days=x) for x in range(1, future_days+1)]
    fig, ax = plt.subplots(figsize=(40, 20))

    # Change the background color to black
    plt.rcParams['figure.facecolor'] = 'black'
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.plot(df.index[display_at:], lstm_model_real_prediction[display_at:], label='LSTM Prediction', color='magenta', linewidth=5.0)
    ax.plot(predicted_dates, lstm_model_predict_prediction, label='Forecast', color='aqua', linewidth=5.0)
    ax.plot(df.index[display_at:], df['adj_close'][display_at:], label='Actual', color='lightgreen', linewidth=5.0)

    # Format the x-axis dates
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)

    plt.legend(prop={'size': 35})  # Increase the size of the legend
    plt.xticks(fontsize=30)  # Increase x-axis font size
    plt.yticks(fontsize=30)  # Increase y-axis font size
    plt.show()
    
    
    
# model, evaluations, df = run_model(
#     file_name='yhoofinance-daily-historical-data/TSLA_daily_data.csv', 
#     ma_window=5, 
#     lstm_units=50, 
#     dropout=0.2, 
#     epochs=100, 
#     batch_size=32, 
#     test_size=0.2, 
#     future_days=30, 
#     rmse=True, 
#     mse=True, 
#     mape=True, 
#     display_at=0
# )





# In[62]:


def main():
    st.title('LSTM Stock Price Prediction')
     
    st.sidebar.markdown('# Parameters')

    # Get a list of all the stock symbols in the 'yhoofinance-daily-historical-data/' folder
    stock_symbols = list_files_in_s3_bucket('raw-stock-price', 'yhoofinance-daily-historical-data/')

    # Use this list to populate the dropdown menu
    stock_symbol = st.sidebar.selectbox('Stocks', stock_symbols)

    # Construct the file name from the selected stock symbol
    file_name = f'yhoofinance-daily-historical-data/{stock_symbol}_daily_data.csv'
    ma_window = st.sidebar.slider('Moving Avg. -- Window Size', 1, 100, 50)
    lstm_units = st.sidebar.slider('LSTM Units', 10, 200, 50)
    dropout = st.sidebar.slider('Dropout', 0.1, 0.9, 0.2)
    epochs = st.sidebar.slider('Epochs', 10, 200, 100)
    batch_size = st.sidebar.slider('Batch Size', 1, 64, 32)
    test_size = st.sidebar.slider('Test Set Size', 0.1, 0.9, 0.2)
    future_days = st.sidebar.slider('Days to Forecast', 1, 50, 30)
    display_at = st.sidebar.slider('Display From Day', 0, 365, 0)

    features = st.sidebar.multiselect('Features', options=['MA', 'adj_close'], default=['adj_close'])

    metrics = st.sidebar.multiselect('Evaluation Metrics', options=['RMSE', 'MSE', 'MAPE'], default=['RMSE', 'MSE', 'MAPE'])

    rmse = 'RMSE' in metrics
    mse = 'MSE' in metrics
    mape = 'MAPE' in metrics

    if st.sidebar.button('Train Model'):
        st.markdown('## Training Model...')

        model, evaluations, df = run_model(
            file_name=file_name,
            ma_window=ma_window,
            lstm_units=lstm_units,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            test_size=test_size,
            future_days=future_days,
            rmse=rmse,
            mse=mse,
            mape=mape,
            display_at=display_at
        )

        # Display evaluation metrics in multiple columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("RMSE")
            st.write(evaluations['rmse'])

        with col2:
            st.header("MSE")
            st.write(evaluations['mse'])

        with col3:
            st.header("MAPE")
            st.write(evaluations['mape'])

        st.markdown('## Forecast Plot')
        st.pyplot()

        if model is not None:
             st.markdown('')
        else:
            st.markdown('## An error occurred during model training')

if __name__ == '__main__':
    main()


# In[68]:


if __name__ == "__main__":
    # Set your file path
    file_name = 'yhoofinance-daily-historical-data/TSLA_daily_data.csv'
    
    # Set your parameters
    ma_window = 5
    lstm_units = 50
    dropout = 0.2
    epochs = 100
    batch_size = 32
    test_size = 0.2
    future_days = 30
    rmse = True
    mse = True
    mape = True
    display_at = 0

    # Run the model
    model, evaluations, df = run_model(
        file_name=file_name,
        ma_window=ma_window,
        lstm_units=lstm_units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        test_size=test_size,
        future_days=future_days,
        rmse=rmse,
        mse=mse,
        mape=mape,
        display_at=display_at
    )

    # Print the evaluations
    print(evaluations)


# In[ ]:





# In[ ]:




