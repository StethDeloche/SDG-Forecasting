import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Test data - simulating the Albania poverty data
dates = pd.date_range(start='2002-01-01', periods=21, freq='A')
poverty_values = [25.4, 23.8, 21.2, 19.6, 18.1, 16.8, 15.2, 14.1, 13.5, 12.8, 12.3, 11.9, 11.5, 11.1, 10.8, 10.4, 10.1, 9.8, 9.5, 9.2, 8.9]

# External variables (simplified)
gdp_values = [4348068242, 8052077248, 12881352893, 12319834195, 13228147516, 11386853113, 11861199830, 13019726211, 15379509891, 15379509891, 15379509891, 15379509891, 15379509891, 15379509891, 15585105131, 15241458744, 15241458744, 15241458744, 15241458744, 15241458744, 15241458744]
unemployment_values = [4.615, 6.566, 13.06, 13.376, 18.055, 17.193, 15.418, 13.616, 12.304, 12.304, 12.304, 12.304, 12.304, 12.304, 11.466, 11.639, 11.639, 11.639, 11.639, 11.639, 11.639]

# Create series
series = pd.Series(poverty_values, index=dates)
print(f"Series shape: {series.shape}")

# Create external variables array (normalize large values)
gdp_normalized = [(x - min(gdp_values)) / (max(gdp_values) - min(gdp_values)) for x in gdp_values]
unemployment_normalized = [(x - min(unemployment_values)) / (max(unemployment_values) - min(unemployment_values)) for x in unemployment_values]

exog_array = np.column_stack([gdp_normalized, unemployment_normalized])
print(f"External variables shape: {exog_array.shape}")
print(f"GDP variation: {np.std(gdp_normalized):.6f}")
print(f"Unemployment variation: {np.std(unemployment_normalized):.6f}")

# Split data
train_size = int(len(series) * 0.8)
train_series = series[:train_size]
test_series = series[train_size:]
train_exog = exog_array[:train_size]
test_exog = exog_array[train_size:]

print(f"Training data: {len(train_series)} points, Test data: {len(test_series)} points")

# Try different ARIMA orders
orders_to_try = [(1,1,1), (1,1,0), (0,1,1), (1,0,1), (1,0,0)]

for order in orders_to_try:
    try:
        print(f"\nTrying SARIMAX with order {order}")
        
        # Fit model on training data for evaluation
        eval_model = SARIMAX(train_series, exog=train_exog, order=order)
        eval_model_fit = eval_model.fit(maxiter=50, disp=False)
        
        # Make predictions for test period
        if len(test_series) > 0:
            predictions = eval_model_fit.forecast(steps=len(test_series), exog=test_exog)
            rmse = np.sqrt(mean_squared_error(test_series, predictions))
            print(f"✓ SARIMAX {order} evaluation RMSE: {rmse:.4f}")
            
            # If successful, fit on all data
            full_model = SARIMAX(series, exog=exog_array, order=order)
            full_model_fit = full_model.fit(maxiter=50, disp=False)
            print(f"✓ Full model fitted successfully with {order}")
            break
        
    except Exception as order_error:
        print(f"✗ SARIMAX {order} failed: {str(order_error)[:100]}")
        continue
else:
    print("\n❌ All SARIMAX orders failed") 