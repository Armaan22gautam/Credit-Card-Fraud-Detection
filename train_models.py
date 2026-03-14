import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import pickle
import os

from data_preprocessing import load_data, preprocess_data

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Trains and evaluates L.R., KNN, SVM, and D.T."""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        # SVM is extremely slow on this large dataset (284k records), disabling for now.
        # 'SVM': SVC(probability=True, kernel='linear') 
    }
    
    results = {}
    
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('reports/figures'):
        os.makedirs('reports/figures')
        
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        with open(f'models/{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'roc_curve': roc_curve(y_test, y_prob),
            'pr_curve': precision_recall_curve(y_test, y_prob)
        }
        
        # Print results
        print(f"\n--- {name} Results ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {results[name]['roc_auc']:.4f}")

    return results

def plot_comparisons(results, output_dir='reports/figures'):
    """Generates comparison plots for all models."""
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, res in results.items():
        fpr, tpr, _ = res['roc_curve']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig(f'{output_dir}/roc_comparison.png')
    plt.close()
    
    # 2. Metric Comparison Bar Chart
    metrics = ['precision', 'recall', 'f1-score']
    model_names = list(results.keys())
    
    # Extracting metrics for fraud class (Class 1)
    plot_data = []
    for m in metrics:
        for name in model_names:
            score = results[name]['report']['1'][m]
            plot_data.append({'Model': name, 'Metric': m.capitalize(), 'Score': score})
            
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot, palette='magma')
    plt.title('Model Performance Comparison (Fraud Class)')
    plt.ylim(0, 1.1)
    plt.savefig(f'{output_dir}/metric_comparison.png')
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    try:
        data = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        # Scaling down for demonstration if CPU/Memory is an issue (Optional)
        # X_train = X_train[:50000]
        # y_train = y_train[:50000]
        
        results = train_and_evaluate(X_train, X_test, y_train, y_test)
        plot_comparisons(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
