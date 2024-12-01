import numpy as np
import pandas as pd
# import pandas_ta as ta

# import talib as ta
import finta as TA

import logging
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv
import joblib

# For modeling
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import optuna

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Load environment variables
load_dotenv()

# Configuration from .env or command line
crypto_name = os.getenv('CRYPTO', 'BTC')
base_timeframe = os.getenv('BASE_TIMEFRAME', '4h')
timeframes = os.getenv('TIMEFRAMES', '1h,4h,1d,1w').split(',')
input_dir = os.getenv('INPUT_DIR', './files/crypto')
output_dir = os.getenv('OUTPUT_DIR', './files/model')

def load_data(crypto_name, timeframes, input_dir):
    data_dict = {}
    for tf in timeframes:
        filename = f"{crypto_name}.{tf}.csv"
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            df['timestamp'] = pd.to_datetime(df['unix'], unit='s')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'Volume BTC']]
            df.rename(columns={'Volume BTC': 'volume'}, inplace=True)
            df.sort_values('timestamp', inplace=True)
            data_dict[tf] = df
            logging.info(f"Loaded data for {crypto_name} timeframe {tf}")
        except FileNotFoundError:
            logging.warning(f"File {filename} not found in {input_dir}. It may need to be generated.")
        except pd.errors.EmptyDataError:
            logging.error(f"No data: {filename} is empty.")
        except pd.errors.ParserError:
            logging.error(f"Parsing error in {filename}.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {filename}: {e}")
    return data_dict

def generate_missing_timeframes(data_dict, timeframes):
    """
    Generate missing timeframes using available lower-level data.
    
    Parameters:
        data_dict (dict): Dictionary of existing dataframes keyed by timeframe.
        timeframes (list): List of timeframes to consider.
    
    Returns:
        dict: Updated dictionary with generated timeframes added.
    """
    # Define a mapping from lower to higher timeframe
    resample_mapping = {
        '1h': '4h',
        '4h': '1d',
        '1d': '1w'
    }
    
    for lower_tf, higher_tf in resample_mapping.items():
        if higher_tf in timeframes and higher_tf not in data_dict and lower_tf in data_dict:
            logging.info(f"Generating missing {higher_tf} data from {lower_tf} data.")
            df = data_dict[lower_tf]
            resampled_df = resample_dataframe(df, higher_tf)
            data_dict[higher_tf] = resampled_df
            logging.info(f"Generated {higher_tf} data from {lower_tf} successfully.")
    
    return data_dict

def resample_dataframe(df, target_timeframe):
    """
    Resample the given DataFrame to the target timeframe.
    
    Parameters:
        df (DataFrame): The input DataFrame with 'timestamp' and OHLCV data.
        target_timeframe (str): The target timeframe to resample to.
    
    Returns:
        DataFrame: The resampled DataFrame.
    """
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)

    if target_timeframe == '4h':
        rule = '4h'
    elif target_timeframe == '1d':
        rule = 'D'
    elif target_timeframe == '1w':
        rule = 'W'
    else:
        logging.error(f"Unsupported target timeframe: {target_timeframe}")
        return df_copy

    resampled_df = df_copy.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()

    return resampled_df



# def calculate_indicators(df):
#     """
#     Calculates technical indicators and adds them to the DataFrame.

#     Parameters:
#         df (DataFrame): DataFrame containing at least the 'close' price column.

#     Returns:
#         DataFrame: The original DataFrame with new indicator columns added.
#     """
#     try:
#         # Moving Averages
#         df['MA20'] = df['close'].rolling(window=20).mean()
#         df['MA50'] = df['close'].rolling(window=50).mean()
        
#         # Relative Strength Index (RSI)
#         delta = df['close'].diff()
#         up = delta.clip(lower=0)
#         down = -1 * delta.clip(upper=0)
#         period = 14
#         gain = up.rolling(window=period).mean()
#         loss = down.rolling(window=period).mean()
#         RS = gain / loss
#         df['RSI'] = 100.0 - (100.0 / (1.0 + RS))

#         # Adding Symbolic Features
#         df['is_oversold'] = np.where(df['RSI'] < 30, 1, 0)
#         df['is_overbought'] = np.where(df['RSI'] > 70, 1, 0)
        
#         # Moving Average Convergence Divergence (MACD)
#         exp1 = df['close'].ewm(span=12, adjust=False).mean()
#         exp2 = df['close'].ewm(span=26, adjust=False).mean()
#         df['MACD'] = exp1 - exp2

#         logging.info("Calculated technical indicators.")
#         return df
#     except Exception as e:
#         logging.error(f"Error calculating indicators: {e}")
#         return df


# import pandas as pd
# # import pandas_ta as ta
# import logging

# from finta import TA
# import pandas as pd
# import logging

def calculate_indicators(df):
    """
    Calculates technical indicators and adds them to the DataFrame using Finta.

    Parameters:
        df (DataFrame): DataFrame containing at least the 'close', 'high', and 'low' price columns.

    Returns:
        DataFrame: The original DataFrame with new indicator columns added.
    """
    try:
        # Validate input
        required_columns = {'close', 'high', 'low'}
        if not required_columns.issubset(df.columns):
            logging.error(f"Input DataFrame must contain the following columns: {required_columns}")
            return df

        # Add Moving Averages
        df['MA20'] = TA.SMA(df, 20)  # 20-period Simple Moving Average
        df['MA50'] = TA.SMA(df, 50)  # 50-period Simple Moving Average

        # Add RSI
        df['RSI'] = TA.RSI(df, 14)  # 14-period Relative Strength Index

        # Add Overbought and Oversold Flags
        df['is_oversold'] = (df['RSI'] < 30).astype(int)
        df['is_overbought'] = (df['RSI'] > 70).astype(int)

        # Add MACD
        macd = TA.MACD(df)  # MACD is returned as a dictionary
        df['MACD'] = macd['MACD']  # MACD line
        df['Signal_Line'] = macd['SIGNAL']  # Signal line
        df['MACD_Histogram'] = macd['MACD'] - macd['SIGNAL']  # Histogram (difference between MACD and Signal line)

        # Add Bullish and Bearish Signals
        df['bullish_signal'] = ((df['MACD'] > df['Signal_Line']) & (df['MACD'] > 0)).astype(int)
        df['bearish_signal'] = ((df['MACD'] < df['Signal_Line']) & (df['MACD'] < 0)).astype(int)

        # Add ROC (Rate of Change)
        df['ROC'] = TA.ROC(df, 12)  # 12-period Rate of Change

        # Add ATR (Average True Range)
        df['ATR'] = TA.ATR(df, 14)  # 14-period Average True Range

        # Add Numeric Trend
        df['trend'] = (df['MA20'] > df['MA50']).astype(int) * 2 - 1  # 1 for bullish, -1 for bearish

        df['PCP'] = df['close'].pct_change() * 100
        df['PPO'] = ((df['EMA20'] - df['EMA50']) / df['EMA50']) * 100

        # Fill NaN values (optional)
        df.fillna(method='bfill', inplace=True)

        logging.info("Calculated technical indicators successfully.")
        return df

    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return df



def merge_timeframes(data_dict, base_tf):
    """
    Merges DataFrames from different timeframes into a single DataFrame aligned on the base timeframe.

    Parameters:
        data_dict (dict): Dictionary of DataFrames keyed by timeframe.
        base_tf (str): The base timeframe to align data on.

    Returns:
        DataFrame: Merged DataFrame containing data from all timeframes.
    """
    try:
        base_df = data_dict[base_tf].copy()
        base_df = base_df.set_index('timestamp')
        
        for tf, df in data_dict.items():
            if tf != base_tf:
                df = df.set_index('timestamp')
                # Resample or align to base timeframe
                df = df.resample(base_tf).ffill()
                suffix = f"_{tf}"
                df = df.add_suffix(suffix)
                base_df = base_df.join(df, how='left')
                logging.info(f"Merged data from timeframe {tf}")
        
        base_df.reset_index(inplace=True)
        # Remove rows with NaN values after merging
        base_df.dropna(inplace=True)
        return base_df
    except KeyError as e:
        logging.error(f"Key error during merging timeframes: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during merging timeframes: {e}")
        return None

def prepare_dataset(df):
    """
    Prepares the dataset by defining the target variable and splitting into features and labels.

    Parameters:
        df (DataFrame): The merged DataFrame containing all features.

    Returns:
        tuple: A tuple containing the features DataFrame (X) and the labels Series (y).
    """
    try:
        if 'close' not in df.columns:
            logging.error("The 'close' column is missing from the DataFrame.")
            return None, None

        # Define the target variable with three classes (-1, 0, 1)
        df['return'] = df['close'].pct_change().shift(-1)
        df['target'] = df['return'].apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Features and Labels
        X = df.drop(['timestamp', 'target', 'return'], axis=1)
        y = df['target']
        logging.info("Prepared dataset with features and target variable (0, 1, 2).")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")
        return None, None


def split_data(X, y, test_size=0.2):
    """
    Splits the dataset into training and testing sets while preserving the temporal order.

    Parameters:
        X (DataFrame): The features DataFrame.
        y (Series): The labels Series.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing training and testing sets for features and labels.
    """
    try:
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        logging.info(f"Split data into training and testing sets. Training size: {len(X_train)}, Testing size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return None, None, None, None

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.

    Parameters:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.

    Returns:
        tuple: Scaled training and testing features, and the scaler object.
    """
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Scaled feature data.")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        return None, None, None

# def train_model(X_train, y_train):
#     """
#     Trains the XGBoost classifier using GridSearchCV for hyperparameter tuning.

#     Parameters:
#         X_train (array): Scaled training features.
#         y_train (Series): Training labels.

#     Returns:
#         model: The best trained model.
#     """
#     try:
#         weight = 20
#         xgb = XGBClassifier(
#             # objective='multi:softmax',  # Multi-class classification
#             objective='multi:softprob',  # Multi-class classification
#             num_class=3,  # Three classes: 0, 1, 2
#             eval_metric='mlogloss',
#             use_label_encoder=False,
#             scale_pos_weight={0: weight, 1: 1, 2: weight+5}  # Assign higher weights to underrepresented classes
#         )
#         param_grid = {
#             'n_estimators': [50, 100],
#             'max_depth': [3, 5, 7],
#             'learning_rate': [0.01, 0.05, 0.1]
#         }
#         tscv = TimeSeriesSplit(n_splits=5)
#         grid_search = GridSearchCV(
#             estimator=xgb,
#             param_grid=param_grid,
#             cv=tscv,
#             scoring='accuracy',
#             n_jobs=-1
#         )
#         grid_search.fit(X_train, y_train)
#         best_model = grid_search.best_estimator_
#         logging.info(f"Model training complete. Best parameters: {grid_search.best_params_}")
#         return best_model
#     except Exception as e:
#         logging.error(f"Error training model: {e}")
#         return None


# def train_model(X_train, y_train):
#     """
#     Trains the XGBoost classifier using GridSearchCV for hyperparameter tuning.

#     Parameters:
#         X_train (array): Scaled training features.
#         y_train (Series): Training labels.

#     Returns:
#         model: The best trained model.
#     """
#     try:
#         # Log class distribution for reference
#         class_counts = y_train.value_counts()
#         logging.info(f"Class distribution: {class_counts}")

#         # Initialize XGBoost classifier
#         xgb = XGBClassifier(
#             objective='multi:softprob',  # Multi-class classification with probabilities
#             num_class=3,  # Number of classes
#             eval_metric='mlogloss',  # Log loss for multi-class tasks
#         )

#         # Define an expanded hyperparameter grid
#         param_grid = {
#             'n_estimators': [50, 100, 200],
#             'max_depth': [3, 5, 7],
#             'learning_rate': [0.01, 0.05, 0.1],
#             'subsample': [0.8, 1.0],
#             'colsample_bytree': [0.8, 1.0],
#             'gamma': [0, 1, 5]
#         }

#         # TimeSeriesSplit for cross-validation
#         tscv = TimeSeriesSplit(n_splits=5)

#         # GridSearchCV for hyperparameter tuning
#         grid_search = GridSearchCV(
#             estimator=xgb,
#             param_grid=param_grid,
#             cv=tscv,
#             scoring='f1_macro',  # Use F1 Macro for imbalanced classes
#             n_jobs=-1,
#             verbose=1  # Show detailed output for debugging
#         )

#         # Train the model
#         logging.info("Starting GridSearchCV...")
#         grid_search.fit(X_train, y_train)

#         # Retrieve the best model
#         best_model = grid_search.best_estimator_
#         logging.info(f"Model training complete. Best parameters: {grid_search.best_params_}")
#         logging.info(f"Best F1 Macro Score: {grid_search.best_score_:.4f}")
#         return best_model

#     except Exception as e:
#         logging.error(f"Error training model: {e}")
#         return None



def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and logs the performance metrics.

    Parameters:
        model: The trained model.
        X_test (array): Scaled testing features.
        y_test (Series): Testing labels.
    """
    try:
        y_pred = model.predict(X_test)
        
        # Detailed metrics
        precision = precision_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        recall = recall_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
        
        logging.info("Model evaluation results:")
        logging.info(f"Precision per class: {precision}")
        logging.info(f"Recall per class: {recall}")
        logging.info(f"F1-score per class: {f1}")
        
        # Weighted F1-score
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        logging.info(f"Weighted F1-score: {weighted_f1}")
        
        # Classification Report and Confusion Matrix
        logging.info("\n" + classification_report(y_test, y_pred, target_names=['Sell (0)', 'Do Nothing (1)', 'Buy (2)']))
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")


from imblearn.over_sampling import SMOTE

def resample_data(X_train, y_train):
    """
    Applies SMOTE to balance the classes in the training set.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.

    Returns:
        tuple: Resampled training features and labels.
    """
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logging.info("Applied SMOTE to balance the classes in the training set.")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error during resampling: {e}")
        return X_train, y_train


def save_model(model, scaler, output_dir, filename='trained_model.pkl'):
    """
    Saves the trained model and scaler to the specified output directory.

    Parameters:
        model: The trained model to save.
        scaler: The scaler object used for feature scaling.
        output_dir (str): Directory where the model will be saved.
        filename (str): Name of the file to save the model as.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        joblib.dump({'model': model, 'scaler': scaler}, filepath)
        logging.info(f"Model saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def validate_inputs(crypto_name, timeframes, base_timeframe, input_dir, output_dir):
    """
    Validates input parameters to ensure they meet expected conditions.

    Parameters:
        crypto_name (str): Name of the cryptocurrency.
        timeframes (list): List of timeframes.
        base_timeframe (str): The base timeframe for alignment.
        input_dir (str): Directory containing input files.
        output_dir (str): Directory to save output files.

    Returns:
        bool: True if all inputs are valid, False otherwise.
    """
    # Validate crypto_name
    if not isinstance(crypto_name, str) or not crypto_name:
        logging.error("Invalid crypto name provided.")
        return False

    # Validate timeframes
    valid_timeframes = {'1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'}
    if not set(timeframes).issubset(valid_timeframes):
        logging.error("Invalid timeframes provided.")
        return False

    # Validate base_timeframe
    if base_timeframe not in timeframes:
        logging.error("Base timeframe must be one of the provided timeframes.")
        return False

    # Validate directories
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return False
    if not os.path.isdir(output_dir):
        logging.info(f"Output directory '{output_dir}' does not exist. It will be created.")
        os.makedirs(output_dir, exist_ok=True)

    logging.info("Input validation passed.")
    return True

def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train a predictive model for cryptocurrency trading signals.')
    parser.add_argument('--crypto', type=str, required=True, help='Name of the cryptocurrency (e.g., BTC)')
    parser.add_argument('--base_timeframe', type=str, required=True, help='Base timeframe for prediction (e.g., 4h)')
    parser.add_argument('--timeframes', type=str, required=True, nargs='+', help='List of timeframes to use (e.g., 1h 4h 1d)')
    parser.add_argument('--input_dir', type=str, default='./files/crypto', help='Directory containing input CSV files')
    parser.add_argument('--output_dir', type=str, default='./files/model', help='Directory to save the trained model')

    args = parser.parse_args()
    return args

def add_lag_features(df, columns, lags):
    """
    Adds lag features to the DataFrame for the specified columns.

    Parameters:
        df (DataFrame): The original DataFrame.
        columns (list): List of column names for which lag features are to be created.
        lags (int): Number of lags to add for each specified column.

    Returns:
        DataFrame: The DataFrame with lag features added.
    """
    try:
        for column in columns:
            for lag in range(1, lags + 1):
                df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        logging.info(f"Added lag features for columns: {columns} with {lags} lags each.")
        return df
    except Exception as e:
        logging.error(f"Error adding lag features: {e}")
        return df

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR) for the given DataFrame.

    Parameters:
        df (DataFrame): DataFrame with columns 'high', 'low', 'close'.
        period (int): The period over which to calculate the ATR.

    Returns:
        DataFrame: The original DataFrame with an additional 'ATR' column.
    """
    try:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = (df['high'] - df['close'].shift()).abs()
        df['low_close'] = (df['low'] - df['close'].shift()).abs()

        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['ATR'] = df['true_range'].rolling(window=period).mean()

        # Drop intermediate columns used in calculation
        df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)

        logging.info(f"Calculated ATR for period: {period}")
        return df
    except Exception as e:
        logging.error(f"Error calculating ATR: {e}")
        return df

def calculate_rate_of_change(df, column, period=14):
    """
    Calculates the Rate of Change (ROC) for the given column in the DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing the data.
        column (str): The column name for which ROC is to be calculated.
        period (int): The period over which to calculate the ROC.

    Returns:
        DataFrame: The original DataFrame with an additional 'ROC' column.
    """
    try:
        df[f'ROC_{column}_{period}'] = df[column].pct_change(periods=period) * 100
        logging.info(f"Calculated Rate of Change (ROC) for column '{column}' over period {period}")
        return df
    except Exception as e:
        logging.error(f"Error calculating Rate of Change (ROC): {e}")
        return df

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna to optimize hyperparameters of XGBClassifier.
    
    Parameters:
        trial (optuna.Trial): A Trial object for suggesting hyperparameter values.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation features.
        y_val (Series): Validation labels.
    
    Returns:
        float: The weighted F1-score on the validation set.
    """
    param = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [{0: 10, 1: 1, 2: 10}, {0: 15, 1: 1, 2: 15}, {0: 20, 1: 1, 2: 20}])
    }
    
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    weighted_f1 = f1_score(y_val, y_pred, average='weighted')
    return weighted_f1

def fine_tune_model(X_train, y_train):
    """
    Fine-tunes the XGBoost model using Optuna for hyperparameter optimization.

    Parameters:
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.

    Returns:
        model: The best trained model.
    """
    try:
        # Split training data into training and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # Create an Optuna study to find the best hyperparameters
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_split, y_train_split, X_val, y_val), n_trials=50)
        
        # Train the final model with the best hyperparameters
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        })
        
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        logging.info(f"Model fine-tuning complete. Best parameters: {best_params}")
        return model
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        return None



def main():
    # args = parse_arguments()

    # crypto_name = args.crypto
    # base_timeframe = args.base_timeframe
    # timeframes = args.timeframes
    # input_dir = args.input_dir
    # output_dir = args.output_dir

    # # Input Validation
    # if not validate_inputs(crypto_name, timeframes, base_timeframe, input_dir, output_dir):
    #     logging.error("Input validation failed. Exiting.")
    #     return



    # Step 1: Load Data
    data_dict = load_data(crypto_name, timeframes, input_dir)

    # Step 2: Generate Missing Timeframes
    data_dict = generate_missing_timeframes(data_dict, timeframes)

    # Check if base timeframe data is loaded and not empty
    if base_timeframe not in data_dict or data_dict[base_timeframe].empty:
        logging.error(f"Base timeframe data '{base_timeframe}' not found or empty. Exiting.")
        return
    
    # Step 2: Calculate Indicators
    for tf in timeframes:
        if tf in data_dict:
            data_dict[tf] = calculate_indicators(data_dict[tf])
            logging.info(f"Calculated indicators for timeframe {tf}")
    
    # Step 3: Merge Data
    merged_df = merge_timeframes(data_dict, base_timeframe)
    if merged_df is None:
        logging.error("Merging data failed. Exiting.")
        return
    
    # merged_df = add_lag_features(merged_df, ['close', 'volume', 'RSI', 'MACD'], lags=3)
    merged_df = add_lag_features(merged_df, ['close_1h'], lags=4)
    merged_df = add_lag_features(merged_df, ['close_4h'], lags=2)
    merged_df = add_lag_features(merged_df, ['close_1d'], lags=2)
    logging.info("Added lag features to the merged DataFrame.")

    # # Step 5: Calculate ATR
    # merged_df = calculate_atr(merged_df)
    # logging.info("Calculated ATR for the merged DataFrame.")
    
    # # Step 6: Calculate Rate of Change (ROC)
    # merged_df = calculate_rate_of_change(merged_df, 'close', period=14)
    # logging.info("Calculated Rate of Change (ROC) for the merged DataFrame.")
    
    
    # Step 4: Prepare Dataset
    X, y = prepare_dataset(merged_df)
    if X is None or y is None:
        logging.error("Dataset preparation failed. Exiting.")
        return
    
    # Step 5: Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    if X_train is None:
        logging.error("Data splitting failed. Exiting.")
        return
    
    # Step 7: Resample Data to Handle Class Imbalance
    X_train, y_train = resample_data(X_train, y_train)
    
    # Step 6: Scale Features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    if X_train_scaled is None:
        logging.error("Feature scaling failed. Exiting.")
        return
    
    # Step 10: Fine-Tune Model
    model = fine_tune_model(X_train_scaled, y_train)
    if model is None:
        logging.error("Model fine-tuning failed. Exiting.")
        return
    
    # Step 8: Evaluate Model
    evaluate_model(model, X_test_scaled, y_test)
    
    # Step 9: Save Model
    model_filename = f'{crypto_name}_{base_timeframe}_model.pkl'
    save_model(model, scaler, output_dir, filename=model_filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"An unhandled exception occurred: {e}")
