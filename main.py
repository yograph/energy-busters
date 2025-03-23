import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Define categorical features
categorical_features = [
    'measurement_hour',
    'measurement_day',
    'measurement_month',
    'measurement_dayofweek',
    'Working_Hours'
]

# Load training data
train = pd.read_csv("C:\\Users\\youse\\Downloads\\train.csv", index_col="ID")
train['measurement_time'] = pd.to_datetime(train['measurement_time'])
train['measurement_time_numeric'] = train['measurement_time'].astype('int64') // 1e9
train['measurement_hour'] = train['measurement_time'].dt.hour
train['measurement_day'] = train['measurement_time'].dt.day
train['measurement_month'] = train['measurement_time'].dt.month
train['measurement_dayofweek'] = train['measurement_time'].dt.dayofweek

# Load test data
test = pd.read_csv("C:\\Users\\youse\\Downloads\\test.csv", index_col="ID")
test['measurement_time'] = pd.to_datetime(test['measurement_time'])
test['measurement_time_numeric'] = test['measurement_time'].astype('int64') // 1e9
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

# Parameters for Savitzky-Golay filter
window_size = 101  # Must be odd and less than the length of the data
poly_order = 2     # Polynomial order for Savitzky-Golay filter

# Apply smoothing and filtering
for col in columns_to_smooth:
    # Apply Savitzky-Golay filter
    train[f'{col}_smoothed'] = savgol_filter(train[col], window_length=window_size, polyorder=poly_order)
    test[f'{col}_smoothed'] = savgol_filter(test[col], window_length=window_size, polyorder=poly_order)

train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], sort=False)

# Handle categorical variables using One-Hot Encoding
combined = pd.get_dummies(combined, columns=categorical_features)

# Prepare feature matrices
columns_to_drop = ['target', 'measurement_time', 'measurement_time_numeric'] + columns_to_smooth + ['is_train']
X = combined[combined['is_train'] == 1].drop(columns=columns_to_drop)
y = combined[combined['is_train'] == 1]['target']
X_test = combined[combined['is_train'] == 0].drop(columns=columns_to_drop + ['target'])

# Handle missing values in X
X = X.fillna(X.mean())

# Handle missing values in X_test
X_test = X_test.fillna(X_test.mean())

# Ensure X and X_test have the same columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential()

# Input layer
model.add(Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)))

# Hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output layer with 'linear' activation for regression
model.add(Dense(1, activation='linear'))

# Compile the model with Mean Squared Error loss
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(
    X_scaled, y,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test_scaled).flatten()

# Check for NaNs in predictions
print("Number of NaNs in predictions:", np.isnan(y_pred).sum())

# Ensure that y_pred has the same index as sample_submission
sample_submission = pd.read_csv("C:\\Users\\youse\\Downloads\\sample_submission.csv", index_col="ID")
y_pred_series = pd.Series(y_pred, index=sample_submission.index)

# Assign predictions to the sample submission DataFrame
sample_submission["target"] = y_pred_series

# Verify that there are no NaN values
print(sample_submission)

# Save the predictions
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"C:\\Users\\youse\\Downloads\\submission_{current_time}.csv"
sample_submission.to_csv(filename)

print("Submission saved successfully:", filename)
