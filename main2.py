import pandas as pd
import numpy as np
import logging
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv
import joblib
from collections import Counter

# For modeling
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
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
    """Load historical cryptocurrency data from CSV files."""
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
            logging.warning(f"File {filename} not found in {input_dir}.")
        except pd.errors.EmptyDataError:
            logging.error(f"No data: {filename} is empty.")
        except pd.errors.ParserError:
            logging.error(f"Parsing error in {filename}.")
        except Exception as e:
            logging.error(f"Unexpected error while loading {filename}: {e}")
    return data_dict


def calculate_indicators(df):
    """Calculate technical indicators and add them to the DataFrame."""
    try:
        # Moving Averages
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        period = 14
        gain = up.rolling(window=period).mean()
        loss = down.rolling(window=period).mean()
        RS = gain / loss
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        df['is_oversold'] = (df['RSI'] < 30).astype(int)
        df['is_overbought'] = (df['RSI'] > 70).astype(int)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # Interaction Features
        df['RSI_MACD_interaction'] = df['RSI'] * df['MACD']

        # Rolling Statistics
        df['rolling_std_close'] = df['close'].rolling(window=5).std()

        logging.info("Calculated technical indicators.")
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return df


def prepare_dataset(df):
    """Prepare the dataset by defining features and target variable."""
    try:
        # Define the target variable
        df['return'] = df['close'].pct_change().shift(-1)
        df['target'] = df['return'].apply(lambda x: 2 if x > 0.01 else (0 if x < -0.01 else 1))
        df.dropna(inplace=True)

        # Separate features and target
        X = df.drop(['timestamp', 'target', 'return'], axis=1)
        y = df['target']
        logging.info("Prepared dataset with features and target variable.")
        return X, y
    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")
        return None, None


def split_data(X, y, test_size=0.2):
    """Split the dataset into training and testing sets."""
    try:
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        logging.info(f"Split data into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return None, None, None, None


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logging.info("Scaled feature data.")
        return X_train_scaled, X_test_scaled, scaler
    except Exception as e:
        logging.error(f"Error scaling features: {e}")
        return None, None, None


def calculate_class_weights(y):
    """Calculate dynamic class weights based on class distribution."""
    class_counts = Counter(y)
    total = sum(class_counts.values())
    return {cls: total / count for cls, count in class_counts.items()}


def fine_tune_model(X_train, y_train, class_weights):
    """Fine-tune the XGBoost model using Optuna."""
    try:
        def objective(trial):
            param = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'scale_pos_weight': class_weights
            }
            model = XGBClassifier(**param)
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            model.fit(X_train_split, y_train_split)
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        best_params.update({'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss', 'use_label_encoder': False})
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        logging.info(f"Model fine-tuning complete. Best parameters: {best_params}")
        return model
    except Exception as e:
        logging.error(f"Error during fine-tuning: {e}")
        return None


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    try:
        y_pred = model.predict(X_test)
        logging.info("Model evaluation results:")
        logging.info("\n" + classification_report(y_test, y_pred, target_names=['Sell (0)', 'Do Nothing (1)', 'Buy (2)']))
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        logging.info(f"Weighted F1-score: {f1_score(y_test, y_pred, average='weighted')}")
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovo')
        logging.info(f"ROC AUC Score: {roc_auc}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")

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

def main():
    """Main function to execute the pipeline."""
    data_dict = load_data(crypto_name, timeframes, input_dir)
    for tf in timeframes:
        if tf in data_dict:
            data_dict[tf] = calculate_indicators(data_dict[tf])
    merged_df = data_dict.get(base_timeframe)
    X, y = prepare_dataset(merged_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    class_weights = calculate_class_weights(y_train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    model = fine_tune_model(X_train_scaled, y_train, class_weights)
    evaluate_model(model, X_test_scaled, y_test)
    save_model(model, scaler, output_dir)


if __name__ == "__main__":
    main()
