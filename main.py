import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import Interp1d
import matplotlib.pyplot as plt

def get_gas_price_model(csv_path):
    # Load and clean data
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Convert dates to days from the first date for calculation
    start_date = df['Date'].min()
    df['Days'] = (df['Date'] - start_date).dt.days
    
    # Create the interpolation function
    # 'linear' is safe, but 'cubic' creates a smoother seasonal curve
    interp_model = interp1d(df['Days'], df['Price'], kind='linear', fill_value="extrapolate")
    
    # Calculate a simple linear trend for the extrapolation
    slope, intercept = np.polyfit(df['Days'], df['Price'], 1)
    
    def estimate_price(target_date_str):
        target_date = pd.to_datetime(target_date_str)
        days_passed = (target_date - start_date).days
        
        # If the date is within the historical range (interpolation)
        if days_passed <= df['Days'].max():
            return float(interp_model(days_passed))
        
        # If the date is in the future (extrapolation)
        else:
            # We combine the linear trend with a seasonal sine wave
            # This accounts for the yearly price fluctuations (365 days)
            trend = intercept + slope * days_passed
            seasonality = np.sin(days_passed * 2 * np.pi / 365.25) * df['Price'].std()
            return float(trend + seasonality)
            
    return estimate_price, df

# --- Execution ---
if __name__ == "__main__":
    try:
        # Initialize the model
        model_fn, original_df = get_gas_price_model('nat_gas_prices.csv')
        
        # Example Test: Estimate for a specific date
        test_date = "2025-02-15"
        price = model_fn(test_date)
        print(f"The estimated price for {test_date} is: ${price:.2f}")

        # Visualization for verification
        all_dates = pd.date_range(start=original_df['Date'].min(), 
                                 end=original_df['Date'].max() + pd.Timedelta(days=365))
        predicted_prices = [model_fn(d) for d in all_dates]

        plt.figure(figsize=(12, 6))
        plt.plot(all_dates, predicted_prices, label='Model Prediction', color='blue')
        plt.scatter(original_df['Date'], original_df['Price'], color='red', label='Actual Data Points')
        plt.title('Natural Gas Price Interpolation & 1-Year Extrapolation')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")