import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
from datetime import datetime

# Make the models first train on some amount of data, then continue
# and train on the previous points to do the next point
# Load training data
train = pd.read_csv("C:\\Users\\youse\\Downloads\\train.csv", index_col="ID")
train['measurement_time'] = pd.to_datetime(train['measurement_time'])
train['measurement_hour'] = train['measurement_time'].dt.hour
train['measurement_day'] = train['measurement_time'].dt.day
train['measurement_month'] = train['measurement_time'].dt.month
train['measurement_dayofweek'] = train['measurement_time'].dt.dayofweek

# Load test data
test = pd.read_csv("C:\\Users\\youse\\Downloads\\test.csv", index_col="ID")
test['measurement_time'] = pd.to_datetime(test['measurement_time'])
test['measurement_hour'] = test['measurement_time'].dt.hour
test['measurement_day'] = test['measurement_time'].dt.day
test['measurement_month'] = test['measurement_time'].dt.month
test['measurement_dayofweek'] = test['measurement_time'].dt.dayofweek

# Merge 'Working_Hours' from extra data
extra = pd.read_csv("C:\\Users\\youse\\Downloads\\Working.csv")
extra['measurement_time'] = pd.to_datetime(extra['measurement_time'])
train = train.merge(extra[['ID', 'Working_Hours']], on='ID', how='left')
test = test.merge(extra[['ID', 'Working_Hours']], on='ID', how='left')

# Define the columns to smooth
columns_to_smooth = [
    'source_1_temperature', 'source_2_temperature',
    'source_3_temperature', 'source_4_temperature',
    'mean_room_temperature', 'sun_radiation_east',
    'sun_radiation_west', 'sun_radiation_south',
    'sun_radiation_north', 'sun_radiation_perpendicular',
    'wind_speed', 'wind_direction', 'clouds'
]
# correlation map, and in a heat map

# Parameters for Savitzky-Golay filter
window_size_sg = 49  # Must be odd and less than the length of the data
poly_order = 7       # Polynomial order for Savitzky-Golay filter

# Apply smoothing and filtering
for col in columns_to_smooth:
    # Apply Savitzky-Golay filter
    train[f'{col}_smoothed'] = savgol_filter(train[col], window_length=window_size_sg, polyorder=poly_order)
    test[f'{col}_smoothed'] = savgol_filter(test[col], window_length=window_size_sg, polyorder=poly_order)

# Prepare categorical features
categorical_features = [
    'measurement_hour', 'measurement_day', 
    'measurement_month', 'measurement_dayofweek', 'Working_Hours'
]

# One-hot encode categorical features
train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)

# Ensure train and test have the same dummy variables
train, test = train.align(test, join='left', axis=1, fill_value=0)

# Define columns to drop
columns_to_drop = ['target', 'measurement_time'] + columns_to_smooth

# Sort data by time
train = train.sort_values('measurement_time').reset_index(drop=True)
test = test.sort_values('measurement_time').reset_index(drop=True)

# Combine train and test data for rolling window
combined_data = pd.concat([train, test], sort=False).reset_index(drop=True)

# Fill missing values
combined_data = combined_data.fillna(combined_data.mean())

# Rolling window parameters
# 7047 to 8808, 1762

window_size = 400  # Adjust based on your data size and computational resources
predictions = []
actuals = []

# Initialize variables
start_index = window_size
end_index = len(combined_data)

# Loop over the data
for i in range(start_index, end_index):
    # Define the rolling window training data
    train_window = combined_data.iloc[i - window_size:i]
    # Check if 'target' is available in the current window
    if 'target' not in train_window.columns:
        continue  # Skip if target is not available

    # Prepare training data
    X_train = train_window.drop(columns=columns_to_drop, errors='ignore')
    y_train = train_window['target']

    # Prepare test instance
    test_instance = combined_data.iloc[[i]]
    X_test_instance = test_instance.drop(columns=columns_to_drop, errors='ignore')

    # Align columns
    X_train, X_test_instance = X_train.align(X_test_instance, join='left', axis=1, fill_value=0)

    # Train the base model
    base_tree = RandomForestRegressor(max_depth=4, random_state=42, criterion='friedman_mse')
    base_tree.fit(X_train, y_train)

    # Predict using the base model
    base_pred = base_tree.predict(X_train)
    base_pred_test = base_tree.predict(X_test_instance)

    # Calculate residuals
    residuals = y_train - base_pred

    # Train the CatBoost model on residuals
    cat_model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=10
    )
    cat_model.fit(X_train, residuals)

    # Predict residuals
    residual_pred_test = cat_model.predict(X_test_instance)

    # 7047 to 8808, 1762
    # Combine base prediction and residual prediction
    final_pred = base_pred_test + residual_pred_test

    # Store the prediction
    predictions.append(final_pred[0])
    print(f"{predictions[-1]},{i}")
    # Store the actual value if available
    if 'target' in test_instance.columns:
        actuals.append(test_instance['target'].values[0])

# If actual values are available, evaluate the model
if actuals:
    rmse = mean_squared_error(actuals, predictions, squared=False)
    print("Rolling Window RMSE:", rmse)

# Prepare submission
# Get IDs from test data
test_ids = test.index.values + 7047
submission = pd.DataFrame({'ID': test_ids, 'target': predictions[-len(test):]})
submission = submission.set_index('ID')

# Save the predictions
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"C:\\Users\\youse\\Downloads\\submission_rolling_{current_time}.csv"
submission.to_csv(filename)
print("Submission saved successfully:", filename)
