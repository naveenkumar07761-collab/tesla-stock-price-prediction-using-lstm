import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def main():
    print("1. Fetching TSLA stock data...")
    ticker = "TSLA"
    # Download data from 2015 to current date
    try:
        data = yf.download(ticker, start="2015-01-01", end="2026-03-01", progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Extract the 'Close' price for the prediction
    # Handle multi-index columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        dataset_close = data.xs('Close', level=0, axis=1)
    else:
        dataset_close = data[['Close']]
        
    dataset = dataset_close.values
    
    print("2. Preprocessing Data...")
    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Train on 80% of the data
    training_data_len = int(np.ceil(len(dataset) * 0.8))
    train_data = scaled_data[0:int(training_data_len), :]
    
    # Create the training datasets. Let's use 60 days of historical data to predict the next day.
    window_size = 60 
    x_train, y_train = [], []
    
    for i in range(window_size, len(train_data)):
        x_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Reshape the data for LSTM which expects 3D input: (samples, time steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    print("3. Building the LSTM Model...")
    model = Sequential()
    # Adding LSTM layers with Dropout to prevent overfitting
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Adding output layers
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("4. Training the Model... (This might take a few minutes)")
    # Train the model
    model.fit(x_train, y_train, batch_size=32, epochs=20)
    
    print("\n5. Evaluating the Model...")
    # Create the testing data set
    test_data = scaled_data[training_data_len - window_size: , :]
    
    x_test = []
    # y_test contains the actual values for evaluation
    y_test = dataset[training_data_len:, :]
    for i in range(window_size, len(test_data)):
        x_test.append(test_data[i-window_size:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    print("\n6. Generating Plot...")
    # Plot the data
    train = dataset_close[:training_data_len].copy()
    valid = dataset_close[training_data_len:].copy()
    
    # Adjust for pandas DataFrame/Series nuances across different library versions
    if isinstance(valid, pd.Series):
         valid = valid.to_frame()
         train = train.to_frame()
    if isinstance(valid.columns, pd.MultiIndex):
        valid[('Predictions', ticker)] = predictions
    else:
        valid['Predictions'] = predictions
        
    plt.figure(figsize=(16, 8))
    plt.title('Tesla (TSLA) Stock Price Prediction Model (LSTM)')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    
    # Plotting Train, Validation, and Predictions appropriately
    if isinstance(train.columns, pd.MultiIndex):
        plt.plot(train[(ticker)])
        plt.plot(valid[[(ticker), ('Predictions', ticker)]])
    elif 'Close' in train.columns:
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
    else:
         plt.plot(train)
         plt.plot(valid)
         
    plt.legend(['Train', 'Val (Actual)', 'Predictions'], loc='lower right')
    
    # Save the output visualization to a file
    plot_path = 'tesla_prediction_plot.png'
    plt.savefig(plot_path)
    print(f"Plot successfully saved as '{plot_path}' in the current directory.")
    
    try:
        # Attempt to show interactively
        plt.show()
    except Exception as e:
        print("Could not display plot interactively (this is normal in some environments). View the saved image instead.")

if __name__ == '__main__':
    main()
