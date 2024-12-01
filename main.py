import numpy as np
import pandas as pd
# import pandas_ta as ta

# import talib as ta
import finta as fintaTA

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

TA = fintaTA.TA

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
        df['SMA20'] = TA.SMA(df, 20)  # 20-period Simple Moving Average
        df['SMA50'] = TA.SMA(df, 50)  # 50-period Simple Moving Average
        
        # Add Moving Averages
        df['EMA20'] = TA.EMA(df, 20)  # 20-period Simple Moving Average
        df['EMA50'] = TA.EMA(df, 50)  # 50-period Simple Moving Average
        
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
        df['trend'] = (df['SMA20'] > df['SMA50']).astype(int) * 2 - 1  # 1 for bullish, -1 for bearish

        df['PCP'] = df['close'].pct_change() * 100
        df['PPO'] = ((df['EMA20'] - df['EMA50']) / df['EMA50']) * 100

        df['Volatility_Adjusted_ROC'] = df['ROC'] / df['ATR']
        df['Rel_Price_SMA20'] = df['close'] / df['SMA20']
        df['MACD_Price_Divergence'] = df['close'].diff() - df['MACD_Histogram'].diff()
        # df['is_green'] = (df['close'] > df['open']).astype(int)
        df['candle_body_percent'] = abs(df['close'] - df['open']) / (df['high'] - df['low']) * 100


        # Fill NaN values (optional)
        df.fillna(method='bfill', inplace=True)

        logging.info("Calculated technical indicators successfully.")
        return df

    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return df

def add_consecutive_green_candles_feature(df):
    """
    Adds a feature to count how many green candles follow the current one.

    Parameters:
        df (DataFrame): DataFrame containing 'open' and 'close' price columns.

    Returns:
        DataFrame: Original DataFrame with an additional 'consecutive_green' column.
    """
    try:
        # Validate required columns
        required_columns = {'open', 'close'}
        if not required_columns.issubset(df.columns):
            logging.error(f"Input DataFrame must contain the following columns: {required_columns}")
            return df

        # Add a binary column to indicate if the candle is green
        df['is_green'] = (df['close'] > df['open']).astype(int)

        # Calculate the number of consecutive green candles that follow each candle
        consecutive_green = []
        n = len(df)
        for i in range(n):
            count = 0
            for j in range(i + 1, n):
                if df.loc[j, 'is_green'] == 1:
                    count += 1
                else:
                    break
            consecutive_green.append(count)

        df['consecutive_green'] = consecutive_green

        # Drop 'is_green' if not needed for further analysis
        df.drop(columns=['is_green'], inplace=True)

        logging.info("Added 'consecutive_green' feature.")
        return df

    except Exception as e:
        logging.error(f"Error adding 'consecutive_green' feature: {e}")
        return df

def add_predictive_candle_features(df):
    """
    Adds features predicting how many green and red candles will follow the current candle.

    Parameters:
        df (DataFrame): DataFrame containing 'open' and 'close' price columns.

    Returns:
        DataFrame: Original DataFrame with 'future_green' and 'future_red' columns.
    """
    try:
        # Validate required columns
        required_columns = {'open', 'close'}
        if not required_columns.issubset(df.columns):
            logging.error(f"Input DataFrame must contain the following columns: {required_columns}")
            return df

        # Determine if each candle is green (1) or red (-1)
        df['candle_direction'] = (df['close'] > df['open']).astype(int)
        df['candle_direction'] = df['candle_direction'].replace(0, -1)

        # Calculate future green and red candles
        future_green = [0] * len(df)
        future_red = [0] * len(df)

        for i in range(len(df)):
            green_count = 0
            red_count = 0
            for j in range(i + 1, len(df)):
                if df.loc[j, 'candle_direction'] == 1:
                    green_count += 1
                    if red_count > 0:  # Stop counting if trend reverses
                        break
                elif df.loc[j, 'candle_direction'] == -1:
                    red_count += 1
                    if green_count > 0:  # Stop counting if trend reverses
                        break
            future_green[i] = green_count
            future_red[i] = red_count

        df['future_green'] = future_green
        df['future_red'] = future_red

        # Drop the temporary candle_direction column
        df.drop(columns=['candle_direction'], inplace=True)

        logging.info("Added 'future_green' and 'future_red' predictive features.")
        return df

    except Exception as e:
        logging.error(f"Error adding predictive candle features: {e}")
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

def create_signal_optimization_target(df, profit_threshold=0.03, loss_threshold=-0.03, max_horizon=6):
    """
    Creates a target to optimize buy and sell signals based on maximizing profit or minimizing loss.

    Parameters:
        df (DataFrame): DataFrame containing 'close'.
        profit_threshold (float): Minimum percentage profit for a Buy signal.
        loss_threshold (float): Maximum percentage loss for a Sell signal.
        max_horizon (int): Maximum number of candles to look ahead.

    Returns:
        DataFrame: Original DataFrame with an additional 'target' column.
    """
    try:
        # Calculate rolling max and min prices within the future window
        df['future_max'] = df['close'].rolling(window=max_horizon, min_periods=1).max().shift(-1)
        df['future_min'] = df['close'].rolling(window=max_horizon, min_periods=1).min().shift(-1)

        # Calculate future profit and loss
        df['max_return'] = (df['future_max'] - df['close']) / df['close']
        df['min_return'] = (df['future_min'] - df['close']) / df['close']

        # Define the target
        df['target'] = df.apply(
            lambda row: 2 if row['max_return'] >= profit_threshold else
                        (0 if row['min_return'] <= loss_threshold else 1),
            axis=1
        )

        # Drop intermediate columns if no longer needed
        df.drop(columns=['future_max', 'future_min', 'max_return', 'min_return'], inplace=True)

        logging.info("Created signal optimization target.")
        return df

    except Exception as e:
        logging.error(f"Error creating signal optimization target: {e}")
        return df


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
        # df['return'] = df['close'].pct_change().shift(-1)
        # df['target'] = df['return'].apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
        
        df = create_signal_optimization_target(df, profit_threshold=0.03, loss_threshold=-0.03, max_horizon=6)
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Features and Labels
        X = df.drop(['timestamp', 'target'], axis=1)
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
        'objective': 'multi:softprob',
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

def get_top_features(model, feature_names, top_n=20):
    """
    Returns the top N most relevant features based on feature importance.

    Parameters:
        model: Trained XGBoost model.
        feature_names (list): List of feature names corresponding to the feature matrix.
        top_n (int): Number of top features to return.

    Returns:
        list: A list of tuples with feature names and their importance scores.
    """
    try:
        # Get feature importance scores
        importance = model.feature_importances_
        
        # Create a DataFrame for better sorting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
        
        # Extract the top N features as a list of tuples
        top_features = feature_importance_df.head(top_n).to_records(index=False)
        top_features_list = [(row['Feature'], row['Importance']) for row in top_features]
        
        logging.info(f"Retrieved top {top_n} features.")
        return top_features_list
    except Exception as e:
        logging.error(f"Error retrieving top features: {e}")
        return []


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
            data_dict[tf] = add_predictive_candle_features(data_dict[tf])
            logging.info(f"Calculated indicators for timeframe {tf}")
    
    # Step 3: Merge Data
    merged_df = merge_timeframes(data_dict, base_timeframe)
    if merged_df is None:
        logging.error("Merging data failed. Exiting.")
        return
    
    # merged_df = add_lag_features(merged_df, ['close', 'volume', 'RSI', 'MACD'], lags=3)
    merged_df = add_lag_features(merged_df, ['close'], lags=4)
    merged_df = add_lag_features(merged_df, ['close_4h'], lags=2)
    merged_df = add_lag_features(merged_df, ['close_1d'], lags=2)
    merged_df['future_green_ratio'] = merged_df['future_green'] / merged_df['future_green_4h']
    merged_df['future_red_ratio'] = merged_df['future_red'] / merged_df['future_red_4h']

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
    if model is not None:
        # Get the top features as a list
        top_features = get_top_features(model, X.columns, top_n=20)
        
        # Display the list of features
        for rank, (feature, importance) in enumerate(top_features, start=1):
            print(f"{rank}. {feature}: {importance:.4f}")
        
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
