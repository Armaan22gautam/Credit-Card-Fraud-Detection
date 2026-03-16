import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def load_data(file_path='data/creditcard.csv'):
    """Loads the dataset from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download it from Kaggle and place it in the data/ directory.")
    
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocesses the data: scaling and handling imbalance."""
    # Scaling 'Amount' and 'Time' (other features are already PCA transformed)
    scaler_amount = StandardScaler()
    scaler_time = StandardScaler()
    df['scaled_amount'] = scaler_amount.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler_time.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Dropping original columns
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Reordering columns to put 'Class' at the end
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handling class imbalance with SMOTE (on training data only)
    print("Original training class distribution:", y_train.value_counts().to_dict())
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("Resampled training class distribution:", pd.Series(y_train_res).value_counts().to_dict())
    
    return X_train_res, X_test, y_train_res, y_test

if __name__ == "__main__":
    try:
        data = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(data)
        print("Data preprocessing complete.")
    except Exception as e:
        print(f"Error: {e}")
