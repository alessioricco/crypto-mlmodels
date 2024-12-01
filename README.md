# Predictive Model for Cryptocurrency Trading Signals

This project involves creating a predictive model for generating trading signals (long or short) for a specific cryptocurrency using multiple timeframes. The model leverages historical OHLCV data and technical indicators such as MACD, RSI, and moving averages.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setting Up the Environment](#setting-up-the-environment)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)
  - [4. Install Required Packages](#4-install-required-packages)
- [Preparing the Data](#preparing-the-data)
  - [1. Directory Structure](#1-directory-structure)
  - [2. Data Requirements](#2-data-requirements)
  - [3. Placing the Data Files](#3-placing-the-data-files)
- [Using the Script](#using-the-script)
  - [1. Script Overview](#1-script-overview)
  - [2. Command-Line Arguments](#2-command-line-arguments)
  - [3. Running the Script](#3-running-the-script)
- [Example Usage](#example-usage)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

- Python 3.6 or higher
- Git (optional, for cloning the repository)

## Setting Up the Environment

Follow these steps to set up your development environment.

### 1. Clone the Repository

Clone the project repository to your local machine (or download the script file if not using Git):

```bash
git clone https://github.com/your_username/crypto-predictive-model.git
cd crypto-predictive-model
```

### 2. Create a Virtual Environment

Create a virtual environment to manage project dependencies without affecting your global Python installation.

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

Activate the virtual environment you just created.

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

- **On macOS and Linux:**

  ```bash
  source venv/bin/activate
  ```

Your command prompt should now indicate that you're working inside the `venv` environment.

### 4. Install Required Packages

Install the necessary Python packages using `pip`.

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```
pandas
numpy
xgboost
scikit-learn
joblib
```

---

## Preparing the Data

### 1. Directory Structure

Ensure your project directory has the following structure:

```
crypto-predictive-model/
├── files/
│   ├── crypto/       # Directory for input CSV files
│   └── model/        # Directory where the trained model will be saved
├── script_name.py    # The main script file
├── requirements.txt
└── README.md
```

### 2. Data Requirements

- **File Naming Convention:** Your CSV files should be named in the format:

  ```
  {crypto_name}.{timeframe}.csv
  ```

  For example:

  ```
  BTC.1h.csv
  BTC.4h.csv
  BTC.1d.csv
  BTC.1w.csv
  ```

- **File Format:** Each CSV file must contain at least the following columns:

  - `timestamp`: The date and time of the data point (in ISO 8601 format or a format parseable by Pandas).
  - `open`, `high`, `low`, `close`, `volume`: OHLCV data.

### 3. Placing the Data Files

Place your CSV data files in the `files/crypto/` directory.

```
crypto-predictive-model/
├── files/
│   ├── crypto/
│   │   ├── BTC.1h.csv
│   │   ├── BTC.4h.csv
│   │   ├── BTC.1d.csv
│   │   └── BTC.1w.csv
│   └── model/
├── script_name.py
├── requirements.txt
└── README.md
```

---

## Using the Script

### 1. Script Overview

The script performs the following steps:

1. **Data Loading:** Reads CSV files for the specified cryptocurrency and timeframes.
2. **Indicator Calculation:** Computes technical indicators (MACD, RSI, moving averages).
3. **Data Merging:** Merges data from different timeframes into a single dataset.
4. **Dataset Preparation:** Defines the target variable and splits data into features and labels.
5. **Data Splitting:** Splits the dataset into training and testing sets while preserving temporal order.
6. **Feature Scaling:** Scales the features using `StandardScaler`.
7. **Model Training:** Trains an XGBoost classifier with hyperparameter tuning using `GridSearchCV`.
8. **Model Evaluation:** Evaluates the model and logs performance metrics.
9. **Model Saving:** Saves the trained model and scaler to the specified output directory.

### 2. Command-Line Arguments

The script accepts the following command-line arguments:

- `--crypto`: **(Required)** Name of the cryptocurrency (e.g., `BTC`).
- `--base_timeframe`: **(Required)** The timeframe to predict on (e.g., `4h`).
- `--timeframes`: **(Required)** List of timeframes to use (e.g., `1h 4h 1d`).
- `--input_dir`: Directory containing input CSV files (default: `./files/crypto`).
- `--output_dir`: Directory to save the trained model (default: `./files/model`).

### 3. Running the Script

Run the script from the project directory with the desired arguments.

```bash
python script_name.py --crypto BTC --base_timeframe 4h --timeframes 1h 4h 1d 1w
```

- **Example with Custom Directories:**

  ```bash
  python script_name.py --crypto ETH --base_timeframe 1d --timeframes 1h 4h 1d 1w --input_dir ./data/crypto --output_dir ./models
  ```

---

## Example Usage

Assuming you have the necessary CSV files for Bitcoin in `files/crypto/`, here's how you would run the script:

```bash
python script_name.py --crypto BTC --base_timeframe 4h --timeframes 1h 4h 1d 1w
```

- This command will:
  - Use Bitcoin (`BTC`) data.
  - Predict on the `4h` timeframe.
  - Use data from `1h`, `4h`, `1d`, and `1w` timeframes.
  - Read input files from `./files/crypto`.
  - Save the trained model to `./files/model/BTC_4h_model.pkl`.

---

## Understanding the Output

- **Logging Information:**
  - The script will log information about each step, including data loading, indicator calculation, data merging, and model training.
  - Logs are displayed in the console. You can configure the script to write logs to a file by modifying the logging configuration.

- **Model File:**
  - The trained model and scaler are saved as a pickle file in the specified output directory.
  - Example: `./files/model/BTC_4h_model.pkl`.

- **Performance Metrics:**
  - The script outputs a classification report and confusion matrix, providing insights into model performance.
  - Metrics include precision, recall, F1-score, and support for each class (long or short signals).

---

## Troubleshooting

- **No Data Found:**
  - Ensure your CSV files are correctly named and placed in the `files/crypto/` directory.
  - Verify that the `timestamp` column exists and is properly formatted in your CSV files.

- **Missing Columns:**
  - Check that all required columns (`timestamp`, `open`, `high`, `low`, `close`, `volume`) are present in your CSV files.

- **Errors During Execution:**
  - Review the console logs for error messages.
  - Ensure all dependencies are installed in your virtual environment.
  - If you encounter a `FileNotFoundError`, verify the input directory and filenames.

- **Invalid Timeframes:**
  - Ensure that the timeframes provided are valid and consistent across your data files.
  - Valid timeframes include: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1M`.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository:**

   Click the "Fork" button at the top right of the repository page to create a copy in your GitHub account.

2. **Clone Your Fork:**

   ```bash
   git clone https://github.com/your_username/crypto-predictive-model.git
   cd crypto-predictive-model
   ```

3. **Create a New Branch:**

   ```bash
   git checkout -b feature/your_feature_name
   ```

4. **Make Changes and Commit:**

   ```bash
   git add .
   git commit -m "Add your commit message"
   ```

5. **Push to Your Fork:**

   ```bash
   git push origin feature/your_feature_name
   ```

6. **Submit a Pull Request:**

   Open a pull request from your fork's branch to the original repository's `main` branch.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note:** Replace `script_name.py`, `your_username`, and other placeholders with the actual script filename, your GitHub username, and any other specific details relevant to your project.

Feel free to reach out if you have any questions or need further assistance!

