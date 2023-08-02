#!/usr/bin/env python
# coding: utf-8

# In[15]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
from utils import comms  # Make sure you have this utils module with db.py file
import boto3
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Set environment variables
access_key = os.getenv('AWS_ACCESS_KEY_ID')
secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS S3 stock bucket
stock_bucket = 'raw-stock-price'
# AWS S3 comment-section bucket
comment_bucket = 'comment-section-st'


def load_data_from_s3(stock_name):
    st.header("Data")
    file_name = f'yhoofinance-daily-historical-data/{stock_name}_daily_data.csv'
    s3 = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)   
    obj = s3.get_object(Bucket=stock_bucket, Key=file_name)
    df = pd.read_csv(obj['Body'])
    df['date'] = pd.to_datetime(df['date'])  # Convert the 'date' column to datetime
    df.set_index('date', inplace=True)  # Set the 'date' column as the index
    df.sort_index(inplace=True)  # Sort the dataframe by date
    st.write(df)

    return df

#@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def train_lstm_model(data, look_back, lstm_units, batch_size, epochs, learning_rate, optimizer, loss):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data['adj_close'].values.reshape(-1,1))

    data_len = len(scaled_data)
    train_len = int(data_len * 0.8)
    train_data = scaled_data[0:train_len]
    test_data = scaled_data[train_len - look_back:]

    X_train, Y_train = create_dataset(train_data, look_back)
    X_test, Y_test = create_dataset(test_data, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], 1), recurrent_regularizer=L2(0.01)))
    model.add(LSTM(units=lstm_units, return_sequences=False, recurrent_regularizer=L2(0.01)))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    if optimizer == 'Adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    return model, scaler, train_data, test_data, X_test, Y_test

def create_dataset(data, look_back):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def plot_predictions(data, train_len, predictions, future_predictions, future_days):
    train = data[:train_len]
    valid = data[train_len:]
    valid['Predictions'] = predictions
    future_dates = pd.date_range(start=valid.index[-1] + pd.DateOffset(days=1), periods=future_days)
    future = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])
    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train['adj_close'])
    plt.plot(valid[['adj_close', 'Predictions']])
    plt.plot(future['Future Predictions'])
    plt.legend(['Train', 'Val', 'Predictions', 'Future'], loc='lower right')
    plt.show()
    st.pyplot()

def calculate_metrics(valid, predictions):
    # Calculate the mean squared error
    mse = mean_squared_error(valid['adj_close'], predictions)

    # Calculate the root mean squared error
    rmse = sqrt(mse)

    # Calculate the mean absolute percentage error
    mape = np.mean(np.abs((valid['adj_close'] - valid['Predictions']) / valid['adj_close'])) * 100

    return mse, rmse, mape

def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
      
    # Display the image using Streamlit with HTML to center it
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col1:
        st.image("assets/futurstox-high-resolution-logo-white-on-transparent-background.png", width=350)

    
    st.title("Stock Price Prediction with LSTM")

    st.sidebar.title("Model Hyperparameters")
    stock_name = st.sidebar.selectbox("Select a stock", ("AAPL", "GOOGL", "MSFT", "AMZN","TSLA","META", "NFLX", "NVDA"), help="Select a stock.")
    learning_rate = st.sidebar.slider("Learning rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, help="Select the learning rate for the model.")
    future_days = st.sidebar.slider("Future days to predict", min_value=1, max_value=30, value=1, step=1, help="Select the number of days to forecast.")


    explanations = st.checkbox('Show Explanations')
    advanced_settings = st.sidebar.checkbox('Advanced Settings')
    if advanced_settings:
        look_back = st.sidebar.slider("Look-back period", min_value=1, max_value=100, value=32, step=1, help="Select the lookback period.")
        lstm_units = st.sidebar.slider("Number of LSTM units", min_value=1, max_value=100, value=32, step=1, help="Select the number of LSTM Units period.")
        batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=100, value=1, step=1, help="Select the batch size.")
        epochs = st.sidebar.slider("Number of epochs", min_value=1, max_value=100, value=1, step=1, help="Select the number of epochs.")
        optimizer = st.sidebar.selectbox("Optimizer", ("Adam", "SGD", "RMSprop"), help="Select the optimizer.")
        loss = st.sidebar.selectbox("Loss function", ("mean_squared_error", "mean_absolute_error", "logcosh"), help="Select the loss function.")

    else:
        look_back = 32
        lstm_units = 32
        batch_size = 32
        epochs = 10
        optimizer = "Adam"
        loss = "mean_squared_error"

            
       

    if st.sidebar.button('Train Model'):
        data = load_data_from_s3(stock_name)


        model, scaler, train_data, test_data, X_test, Y_test = train_lstm_model(data, look_back, lstm_units, batch_size, epochs, learning_rate, optimizer, loss)

        st.write(f"Trained LSTM model for {stock_name} with look-back period = {look_back}, LSTM units = {lstm_units}, batch size = {batch_size}, epochs = {epochs}, learning rate = {learning_rate}, optimizer = {optimizer}, loss function = {loss}")

        # Get the predicted values
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Make predictions for the next future_days days
        input_data = scaler.transform(data['adj_close'].values.reshape(-1,1))[-look_back:]  # start with the last 50 days of data
        future_predictions = []
        for _ in range(future_days):
            pred = model.predict(input_data.reshape(1, -1, 1))  # make a prediction
            future_predictions.append(pred[0, 0])  # store the prediction
            input_data = np.roll(input_data, -1)  # shift the data
            input_data[-1] = pred  # insert the prediction at the end

        # Unscale the predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))

             # Calculate metrics
        valid = data[len(train_data):len(train_data)+len(predictions)]
        valid['Predictions'] = predictions
        mse, rmse, mape = calculate_metrics(valid, predictions)

        # Display evaluation metrics in multiple columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header("MSE")
            st.write(mse)

        with col2:
            st.header("RMSE")
            st.write(rmse)

        with col3:
            st.header("MAPE")
            st.write(mape)

        plot_predictions(data, len(train_data), predictions, future_predictions, future_days)

          # Display explanations on the main page
    if explanations:
        st.markdown('## Explanations')
        st.markdown('**Evaluation Metrics**: Measures used to assess how well the model\'s predictions match the actual values.')
        st.markdown('- **RMSE (Root Mean Squared Error)**: A measure of the differences between the values predicted by the model and the actual values. Smaller values are better, with 0 being a perfect match.')
        st.markdown('- **MSE (Mean Squared Error)**: Similar to RMSE, but without taking the square root. This means larger errors are more heavily penalized.')
        st.markdown('- **MAPE (Mean Absolute Percentage Error)**: The average of the absolute percentage differences between the predicted and actual values. It gives an idea of the error rate in terms of the actual values.')
        st.markdown('- **Learning Rate**:  This is a tuning parameter that determines the step size at each iteration while moving towards a minimum of a loss function. Too big of a learning rate may result going over the global optima. Too small of a learning rate may take a very long time to get to the global optima.')
        if advanced_settings:
            st.markdown('**Look-back Period**: This is the number of previous time steps to use as input variables to predict the next time period. Essentially, it is the window that the model uses to learn and make future predictions.')
            st.markdown('**Number of LSTM Units**: These are the computational units of the LSTM. They are responsible for learning and retaining important dependencies in the data and forgetting the less relevant details. The number of LSTM units is an important parameter and can significantly influence the model\'s ability to extract patterns from data.')
            st.markdown('**Batch Size**: This is the total number of training examples present in a single batch. The model weights are updated after training with each batch. The batch size can significantly impact the model\'s performance and speed of convergence.')
            st.markdown('**Number of Epochs**: An epoch refers to one complete traversal through the entire training dataset. It is a crucial parameter in the context of machine learning and signifies the number of passes of the entire training dataset the machine learning algorithm has completed.')
            st.markdown('**Optimizer**: This is the algorithmic approach employed to adjust the parameters of the LSTM model with the objective of minimizing the loss function. Optimizers such as Adam or Stochastic Gradient Descent are commonly used in this context.')
            st.markdown('**Loss Function**: This is a measure of how well the model\'s predictions conform to the actual values. It is a function that takes the actual and predicted values as input and outputs a numeric value representing the prediction error. The objective of training is to minimize this loss value.')




     # Connect to the S3 bucket
    s3 = comms.connect()
    comment_bucket = 'comment-section-st'
    file_name = 'lstm-st/comments.csv'
    comments = comms.collect(s3, comment_bucket, file_name)

    with st.expander("💬 Open comments"):
        # Show comments
        st.write("**Comments:**")

        for index, entry in enumerate(comments.itertuples()):
            st.markdown(f"**{entry.name}** ({entry.date}):\n\n&nbsp;\n\n&emsp;{entry.comment}\n\n---")


            is_last = index == len(comments) - 1
            is_new = "just_posted" in st.session_state and is_last
            if is_new:
                st.success("☝️ Your comment was successfully posted.")

        # Insert comment
        st.write("**Add your own comment:**")
        form = st.form("comment")
        name = form.text_input("Name")
        comment = form.text_area("Comment")
        submit = form.form_submit_button("Add comment")

        if submit:
            date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            comment.insert(s3, comment_bucket, file_name, [name, comment, date])
            if "just_posted" not in st.session_state:
                st.session_state["just_posted"] = True
            st.experimental_rerun()


if __name__ == "__main__":
    main()

# In[ ]:

