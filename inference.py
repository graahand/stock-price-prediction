import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_input_data(df, window_size):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Convert percentage change to numeric, removing the % sign
    df['Percent Change'] = df['Percent Change'].str.rstrip(' %').astype(float)
    
    # Clean volume data (remove commas and convert to numeric)
    df['Volume'] = df['Volume'].str.replace(',', '').astype(float)

    # Define all numeric columns we want to use as features
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume']
    for col in numeric_columns:
        df[col] = df[col].astype(float)

    # Create additional features
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    
    # Drop rows with NaN values that result from calculating moving averages
    df = df.dropna()

    # Ensure there are enough rows for the window size
    if len(df) <= window_size:
        raise ValueError(f"Dataset must contain more than {window_size} rows, found {len(df)} rows.")

    # Define all features for scaling
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Percent Change', 'Volume', 'MA5', 'MA20']
    
    # Scale the features
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Create sequences (windows of data)
    sequences = []
    for i in range(len(df) - window_size):
        sequences.append(df[feature_columns].iloc[i:i + window_size].values)

    return np.array(sequences), scaler, feature_columns

def predict_from_model(model_path, csv_file, window_size=60, forecast_days=7):
    try:
        # Load the saved model
        model = tf.keras.models.load_model(model_path)

        # Load and preprocess data
        df = pd.read_csv(csv_file)
        
        # Ensure there is enough data for the window size
        if len(df) < window_size:
            raise ValueError(f"Insufficient data. Dataset must have at least {window_size} rows.")
        
        sequences, scaler, feature_columns = preprocess_input_data(df, window_size)

        # Use the last 60 days as input to predict the next 7 days
        last_sequence = sequences[-1:]  # Get the last sequence of 60 days
        
        # Predict the next 'forecast_days' (7 days)
        predictions = []
        for _ in range(forecast_days):
            pred = model.predict(last_sequence)
            predictions.append(pred[0, 0])  # Assuming model predicts only 'Close' price
            
            # Update the sequence with the predicted value for the next step
            next_sequence = np.concatenate((last_sequence[:, 1:, :], pred.reshape(1, 1, -1)), axis=1)
            last_sequence = next_sequence

        predictions = np.array(predictions)

        # Inverse transform predictions back to original scale
        dummy = np.zeros((predictions.shape[0], len(feature_columns)))
        dummy[:, 3] = predictions  # Assuming predictions correspond to 'Close'
        
        predictions_inverse = scaler.inverse_transform(dummy)[:, 3]  # Get 'Close' values
        
        # Prepare for visualization
        dates = pd.to_datetime(df['Date'].values[-forecast_days:])  # Use the last 'forecast_days' dates

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(dates, predictions_inverse, label='Predicted Close Prices', color='red')
        plt.title('Stock Price Prediction for Next Week')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predictions_inverse

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    model_path = "stock_price_predictor.keras"
    csv_file = "test4.csv"
    predict_from_model(model_path, csv_file)
