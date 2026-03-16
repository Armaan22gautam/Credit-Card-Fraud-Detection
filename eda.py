import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='reports/figures'):
    """Performs basic EDA and saves plots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Performing EDA...")
    
    # Class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, hue='Class', palette='viridis', legend=False)
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
    plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/class_distribution.png')
    plt.close()
    
    # Amount distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', kde=True)
    sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='blue', label='Normal', kde=True)
    plt.title('Distribution of Transaction Amount')
    plt.yscale('log') # Log scale because of high variance
    plt.legend()
    plt.savefig(f'{output_dir}/amount_distribution.png')
    plt.close()

    # Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap='coolwarm_r', annot=False)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()

    print(f"EDA plots saved to {output_dir}")

if __name__ == "__main__":
    from data_preprocessing import load_data
    try:
        df = load_data()
        perform_eda(df)
    except Exception as e:
        print(f"Error: {e}")
