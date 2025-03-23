import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import os
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
#time series windowing

# Define the number of folds
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# List files in the Downloads directory
for dirname, _, filenames in os.walk('C:\\Users\\youse\\Downloads'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

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

# Prepare feature matrices
X = train.drop(columns=["target", "measurement_time", "measurement_time_numeric"])
y = train["target"]
X_test = test.drop(columns=["measurement_time", "measurement_time_numeric"])

# Handle missing values in X
nan_columns = X.columns[X.isna().any()].tolist()
for col in nan_columns:
    X[col] = X.groupby('measurement_hour')[col].transform(lambda grp: grp.fillna(grp.mean()))
X = X.fillna(X.mean())

# Handle missing values in X_test
nan_columns_test = X_test.columns[X_test.isna().any()].tolist()
for col in nan_columns_test:
    X_test[col] = X_test.groupby('measurement_hour')[col].transform(lambda grp: grp.fillna(grp.mean()))
X_test = X_test.fillna(X_test.mean())

# Optuna hyperparameter tuning for LightGBM
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'max_bin': trial.suggest_int('max_bin', 255, 1024),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 10.0, log=True),
    }

    rmse_scores = []

    for train_index, valid_index in kf.split(X):
        X_train_fold = X.iloc[train_index]
        X_valid_fold = X.iloc[valid_index]
        y_train_fold = y.iloc[train_index]
        y_valid_fold = y.iloc[valid_index]

        lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
        lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold)

        # Use callbacks for early stopping
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_pred = model.predict(X_valid_fold, num_iteration=model.best_iteration)
        rmse = mean_squared_error(y_valid_fold, y_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=10)
# Check if everything went ok
for trial in study_lgb.trials:
    print(f"Trial {trial.number} - Value: {trial.value} - State: {trial.state}")

# Optuna hyperparameter tuning for XGBoost
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'tree_method': 'gpu_hist',  # Use 'gpu_hist' if using GPU
    }

    rmse_scores = []

    for train_index, valid_index in kf.split(X):
        # Use .copy() to avoid SettingWithCopyWarning
        X_train_fold = X.iloc[train_index].copy()
        X_valid_fold = X.iloc[valid_index].copy()
        y_train_fold = y.iloc[train_index]
        y_valid_fold = y.iloc[valid_index]

        # Ensure categorical features are correctly typed
        for col in categorical_features:
            X_train_fold[col] = X_train_fold[col].astype('category')
            X_valid_fold[col] = X_valid_fold[col].astype('category')

        # Create DMatrix with enable_categorical=True
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold, enable_categorical=True)
        dvalid = xgb.DMatrix(X_valid_fold, label=y_valid_fold, enable_categorical=True)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dvalid, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        y_pred = model.predict(dvalid)
        rmse = mean_squared_error(y_valid_fold, y_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

# remove o niot

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)

for trial in study_xgb.trials:
    print(f"Trial {trial.number} - Value: {trial.value} - State: {trial.state}")

# Optuna hyperparameter tuning for CatBoost
def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'verbose': False,
        # Remove 'task_type': 'GPU' if you don't have a GPU
        # 'task_type': 'GPU'
    }
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)

    # Remove categorical copies of numerical features
    numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols_to_remove = [col + '_cat' for col in numerical_columns]
    X_features = X.drop(columns=['ID', 'ID_cat'] + cat_cols_to_remove, errors='ignore')
    X_test_features = X_test.drop(columns=['ID', 'ID_cat'] + cat_cols_to_remove, errors='ignore')

    # Update categorical_features
    categorical_features = ['measurement_hour', 'measurement_day', 'measurement_month', 'measurement_dayofweek', 'Working_Hours']

    rmse_scores = []

    for train_index, valid_index in kf.split(X_features):
        # Use .copy() to avoid SettingWithCopyWarning
        X_train_fold = X_features.iloc[train_index].copy()
        X_valid_fold = X_features.iloc[valid_index].copy()
        y_train_fold = y.iloc[train_index]
        y_valid_fold = y.iloc[valid_index]

        # Convert categorical features to strings
        for col in categorical_features:
            X_train_fold[col] = X_train_fold[col].astype(str)
            X_valid_fold[col] = X_valid_fold[col].astype(str)

        model = CatBoostRegressor(**params)
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=(X_valid_fold, y_valid_fold),
            cat_features=categorical_features,
            early_stopping_rounds=50,
            verbose=True
        )

        y_pred = model.predict(X_valid_fold)
        rmse = mean_squared_error(y_valid_fold, y_pred, squared=False)
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=10)

# Check if everything went ok
for trial in study_cat.trials:
    print(f"Trial {trial.number} - Value: {trial.value} - State: {trial.state}")

# Generate random seeds
seeds = [np.random.randint(10000) for _ in range(10)]

# Training LightGBM models in parallel
def train_lgb_model(seed):
    params = study_lgb.best_params.copy()
    params['random_state'] = seed
    params['n_estimators'] = params.pop('iterations', 1000)
    params['objective'] = 'regression'
    params['metric'] = 'rmse'
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=categorical_features)
    return model

with Pool(processes=4) as pool:
    lgb_models = pool.map(train_lgb_model, seeds)

# Training XGBoost models in parallel
def train_xgb_model(seed):
    params = study_xgb.best_params.copy()
    params['random_state'] = seed
    params['n_estimators'] = params.pop('iterations', 1000)
    params['objective'] = 'reg:squarederror'
    params['tree_method'] = 'gpu_hist'  # Ensure GPU usage
    
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model

with Pool(processes=4) as pool:
    xgb_models = pool.map(train_xgb_model, seeds)

# Training CatBoost models in parallel
def train_cat_model(seed):
    params = study_cat.best_params.copy()
    params['random_seed'] = seed
    params['cat_features'] = categorical_features
    
    model = CatBoostRegressor(**params)
    model.fit(X, y, verbose=False)
    return model

with Pool(processes=4) as pool:
    cat_models = pool.map(train_cat_model, seeds)

# Training Neural Network models in parallel

# Ensure that TensorFlow doesn't allocate all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Prepare data for NN
X_nn = X.copy()
for col in categorical_features:
    X_nn[col] = X_nn[col].astype('float32')

def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

def train_nn_model(seed):
    model = create_nn_model(X_nn.shape[1])
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model.fit(X_nn, y, epochs=100, batch_size=256, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model

with Pool(processes=4) as pool:
    nn_models = pool.map(train_nn_model, seeds)

# Generate predictions on training data
lgb_preds = np.mean([model.predict(X) for model in lgb_models], axis=0)
xgb_preds = np.mean([model.predict(X) for model in xgb_models], axis=0)
cat_preds = np.mean([model.predict(X) for model in cat_models], axis=0)
nn_preds = np.mean([model.predict(X_nn).flatten() for model in nn_models], axis=0)

preds_df = pd.DataFrame({
    'lgb': lgb_preds.flatten(),
    'xgb': xgb_preds.flatten(),
    'cat': cat_preds.flatten(),
    'nn': nn_preds.flatten()
})

# Train models to improve individual predictions
improved_models = {}
for col in preds_df.columns:
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        verbose=False
    )
    model.fit(preds_df[[col]], y)
    improved_models[col] = model

improved_preds_df = pd.DataFrame()
for col, model in improved_models.items():
    improved_preds_df[col] = model.predict(preds_df[[col]])

# Stacking model
def create_stacking_model(input_dim):
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

stacking_model = create_stacking_model(improved_preds_df.shape[1])
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
stacking_model.fit(improved_preds_df, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

rmse_scores = []

for train_index, valid_index in kf.split(improved_preds_df):
    X_train_fold, X_valid_fold = improved_preds_df.iloc[train_index], improved_preds_df.iloc[valid_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

    model = create_stacking_model(X_train_fold.shape[1])
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, validation_data=(X_valid_fold, y_valid_fold),
              callbacks=[early_stopping], verbose=0)

    y_pred = model.predict(X_valid_fold).flatten()
    rmse = mean_squared_error(y_valid_fold, y_pred, squared=False)
    rmse_scores.append(rmse)

print(f"Mean RMSE: {np.mean(rmse_scores):.5f}, Std RMSE: {np.std(rmse_scores):.5f}")

# Generate predictions on test data
lgb_preds_test = np.mean([model.predict(X_test) for model in lgb_models], axis=0)
xgb_preds_test = np.mean([model.predict(X_test) for model in xgb_models], axis=0)
cat_preds_test = np.mean([model.predict(X_test) for model in cat_models], axis=0)
X_test_nn = X_test.copy()
for col in categorical_features:
    X_test_nn[col] = X_test_nn[col].astype('float32')
nn_preds_test = np.mean([model.predict(X_test_nn).flatten() for model in nn_models], axis=0)

preds_test_df = pd.DataFrame({
    'lgb': lgb_preds_test.flatten(),
    'xgb': xgb_preds_test.flatten(),
    'cat': cat_preds_test.flatten(),
    'nn': nn_preds_test.flatten()
})

improved_preds_test_df = pd.DataFrame()
for col, model in improved_models.items():
    improved_preds_test_df[col] = model.predict(preds_test_df[[col]])

final_predictions = stacking_model.predict(improved_preds_test_df).flatten()

submission = pd.DataFrame({
    'ID': X_test.index,
    'target': final_predictions
})

# Define the directory to save the files
output_dir = r"C:\\Users\\youse\\Downloads"

# Generate a filename with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(output_dir, f"submission_{timestamp}.csv")

# Save the file
submission.to_csv(filename, index=False)
print(f"File saved as: {filename}")
