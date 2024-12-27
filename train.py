"""Python Code to create and stacked LSTM network with 
hyperparameters training for the prediction of stock's 
price, remember, last's day closing price will be same as 
next's day opening price, and the model only need to predict 
closing price of the stock"""

"""Use MinMax scaler for scaling, drop the columns on the basis 
correlation between the closing price and the column, use MAE, MAPE, r-square
Accuracy as the key metrics for assessment and comparative analysis of 
model's performance,  """

"""Deal with the percentage symbol in percent change if it is included in dataset 
after dropping the column based on correlation value, there are '-' sign in the 
percent change column which also need to be dealt with if this column is incorporated"""

"""If possible implement sequential self-attention mechanism along with LSTM 
to develop and hybrid architecture for correct prediction, if possible incorporate
moving average, moving average convergence divergence, Relative strength index 
for robust prediction with technical analysis. """

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, MultiHeadAttention, LayerNormalization
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class StockPricePredictor:
    def __init__(self, window_size=60, lstm_units=50, attention_heads=2):
        self.window_size = window_size
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.scalers = {}
        
    def preprocess_data(self, df):
        try:
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Drop unnecessary columns and handle dates
            df = df.drop(['Percent Change', 'Symbol'], axis=1)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by="Date", ascending=True).reset_index(drop=True)
            
            # Clean Volume column and convert price columns to float
            df['Volume'] = df['Volume'].str.replace(',', '').astype(float)
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # Calculate technical indicators
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Drop Date column before correlation calculation
            df_numeric = df.drop('Date', axis=1)
            
            # Calculate correlations with Close price
            correlations = df_numeric.corr()['Close'].abs().sort_values(ascending=False)
            selected_features = correlations[correlations > 0.3].index.tolist()
            
            print("\nFeature Correlations with Close price:")
            print(correlations)
            print("\nSelected features:", selected_features)
            
            # Select features and scale data
            features = df_numeric[selected_features].values
            for i, feature in enumerate(selected_features):
                self.scalers[feature] = MinMaxScaler()
                features[:, i] = self.scalers[feature].fit_transform(features[:, i].reshape(-1, 1)).ravel()
            
            return features, selected_features
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def create_sequences(self, data):
        try:
            X, y = [], []
            for i in range(len(data) - self.window_size):
                X.append(data[i:(i + self.window_size)])
                y.append(data[i + self.window_size, 0])  # 0 index represents Close price
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"Error in sequence creation: {str(e)}")
            raise
    
    def build_model(self, input_shape):
        try:
            inputs = Input(shape=input_shape)
            
            # LSTM layers with dropout
            lstm_out = LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(inputs)
            lstm_out = LSTM(self.lstm_units, return_sequences=True, dropout=0.2)(lstm_out)
            
            # Self-attention mechanism
            attention_out = MultiHeadAttention(
                num_heads=self.attention_heads, 
                key_dim=self.lstm_units
            )(lstm_out, lstm_out)
            
            # Add & Normalize
            attention_out = LayerNormalization()(attention_out + lstm_out)
            
            # Final LSTM layer
            lstm_out = LSTM(self.lstm_units)(attention_out)
            
            # Output layer
            outputs = Dense(1)(lstm_out)
            
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse')
            return model
        except Exception as e:
            print(f"Error in model building: {str(e)}")
            raise
    
    def plot_predictions(self, y_test, y_pred, title="Stock Price Prediction Results"):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual Prices', color='blue')
        plt.plot(y_pred, label='Predicted Prices', color='red')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_metrics(self, y_true, y_pred):
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            r2 = r2_score(y_true, y_pred)
            return {
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            }
        except Exception as e:
            print(f"Error in metrics calculation: {str(e)}")
            raise
    
    def train_and_evaluate(self, df, train_split=0.8, epochs=50, batch_size=32):
        try:
            # Preprocess data
            features, selected_features = self.preprocess_data(df)
            X, y = self.create_sequences(features)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No sequences created. Check window size and data length.")
            
            # Split data
            train_size = int(len(X) * train_split)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            model = self.build_model((self.window_size, len(selected_features)))
            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Save the trained model
            model.save('stock_price_predictor.keras')
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Inverse transform predictions and actual values
            y_pred = self.scalers['Close'].inverse_transform(y_pred)
            y_test = self.scalers['Close'].inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            
            # Plot results
            self.plot_predictions(y_test, y_pred)
            self.plot_training_history(history)
            
            return model, history, metrics, y_test, y_pred
            
        except Exception as e:
            print(f"Error in training and evaluation: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Read data
        df = pd.read_csv('sanima.csv')
        
        # Initialize and train model
        predictor = StockPricePredictor(window_size=30)  # Reduced window size for smaller datasets
        model, history, metrics, y_test, y_pred = predictor.train_and_evaluate(df, epochs=50)
        
        # Print metrics
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
