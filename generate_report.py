import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

"""
This script generates a full visual report of the entire project,
summarizing all successful training strategies and their results.

It reads no data and is pre-filled with the final metrics from your logs.

It will generate 5 .png files:
1. final_accuracy_progression.png: Bar chart showing accuracy from 79% -> 96%.
2. base_model_comparison.png: Grouped bar chart comparing original models vs. Method 3.
3. meta_model_comparison_v4.png: Bar chart comparing Logistic Regression (88%) vs. XGBoost (96%).
4. final_96_confusion_matrix.png: A heatmap of your best model's confusion matrix.
5. final_96_classification_report.png: A heatmap of your best model's classification report.
"""

# --- Chart 1: Final Accuracy Progression ---
def plot_accuracy_progression():
    print("Generating: 1. Final Accuracy Progression Chart")
    data = {
        'Method': [
            'A: Simple Avg (79.17%)',
            'B: Weighted Avg (83.00%)',
            'C: Stack v1 (LR) (85.00%)',
            'D: Stack v2 (XGB) (89.00%)',
            'E: Stack v3 (XGB) (94.00%)',
            'F: Stack v4 (XGB) (96.00%)'
        ],
        'Accuracy': [79.17, 83.00, 85.00, 89.00, 94.00, 96.00]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='Method', y='Accuracy', data=df, palette='viridis')
    
    ax.set_ylim(75, 100)
    ax.set_title('Final Model Accuracy Progression (Test Set)', fontsize=16, pad=20)
    ax.set_xlabel('Ensemble Strategy', fontsize=12, labelpad=15)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig('final_accuracy_progression.png', dpi=300)
    plt.close()

# --- Chart 2: Base Model (Method 3) Improvement ---
def plot_base_model_comparison():
    print("Generating: 2. Base Model Improvement Chart")
    data = {
        'Backbone': [
            'vgg16', 'densenet121', 'efficientnet_b0', 'resnet18', 'resnet50',
            'vgg16', 'densenet121', 'efficientnet_b0', 'resnet18', 'resnet50'
        ],
        'Training Method': [
            'Original', 'Original', 'Original', 'Original', 'Original',
            'Method 3 (Diff. LR)', 'Method 3 (Diff. LR)', 'Method 3 (Diff. LR)', 
            'Method 3 (Diff. LR)', 'Method 3 (Diff. LR)'
        ],
        'Mean CV Accuracy (%)': [
            68.80, 71.00, 72.20, 73.60, 72.00,
            75.15, 74.75, 75.15, 74.74, 71.34
        ]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(14, 7))
    ax = sns.barplot(x='Backbone', y='Mean CV Accuracy (%)', hue='Training Method', data=df, palette='muted')
    
    ax.set_ylim(65, 80)
    ax.set_title('Base Model Improvement: Original vs. Method 3 (Differential LR)', fontsize=16, pad=20)
    ax.set_xlabel('Backbone Architecture', fontsize=12, labelpad=15)
    ax.set_ylabel('Mean 5-Fold CV Accuracy (%)', fontsize=12)
    plt.legend(title='Training Method', loc='upper right')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.savefig('base_model_comparison.png', dpi=300)
    plt.close()

# --- Chart 3: Meta-Model (v4) Comparison ---
def plot_meta_model_comparison():
    print("Generating: 3. Final Meta-Model Comparison Chart")
    # This is from your v4 run (the 96% one)
    data = {
        'Meta-Model': ['Logistic Regression', 'XGBoost'],
        'Accuracy': [88.00, 96.00]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x='Meta-Model', y='Accuracy', data=df, palette='Set2')
    
    ax.set_ylim(80, 100)
    ax.set_title('Final Meta-Model Performance (on v4 Features)', fontsize=16, pad=20)
    ax.set_xlabel('Model Type', fontsize=12, labelpad=15)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig('meta_model_comparison_v4.png', dpi=300)
    plt.close()

# --- Chart 4: Final 96% Confusion Matrix ---
def plot_final_confusion_matrix():
    print("Generating: 4. Final 96% Confusion Matrix Heatmap")
    # From your 'v4' XGBoost log
    cm_data = [[37, 1, 1],
               [ 1, 29, 0],
               [ 1, 0, 30]]
    
    labels = ['Pre-peak', 'Peak', 'Post-peak']
    
    df_cm = pd.DataFrame(cm_data, index=labels, columns=labels)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', 
                     annot_kws={"size": 16, "weight": "bold"})
    
    ax.set_title('Confusion Matrix - 96.00% Accuracy (XGBoost Stack v4)', fontsize=16, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=15)
    ax.set_ylabel('True Label', fontsize=12, labelpad=15)
    plt.tight_layout()
    plt.savefig('final_96_confusion_matrix.png', dpi=300)
    plt.close()

# --- Chart 5: Final 96% Classification Report ---
def plot_final_classification_report():
    print("Generating: 5. Final 96% Classification Report")
    # From your 'v4' XGBoost log
    report_data = {
        'Pre-peak': {'precision': 0.9487, 'recall': 0.9487, 'f1-score': 0.9487, 'support': 39},
        'Peak': {'precision': 0.9667, 'recall': 0.9667, 'f1-score': 0.9667, 'support': 30},
        'Post-peak': {'precision': 0.9677, 'recall': 0.9677, 'f1-score': 0.9677, 'support': 31},
        'accuracy': {'precision': 0.96, 'recall': 0.96, 'f1-score': 0.96, 'support': 100},
        'macro avg': {'precision': 0.9610, 'recall': 0.9610, 'f1-score': 0.9610, 'support': 100},
        'weighted avg': {'precision': 0.9600, 'recall': 0.9600, 'f1-score': 0.9600, 'support': 100}
    }
    
    df = pd.DataFrame(report_data).iloc[:-1, :].T  # Exclude 'support' row, transpose
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='Greens', vmin=0.9, vmax=1.0)
    plt.title('Classification Report - 96.00% Accuracy (XGBoost Stack v4)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('final_96_classification_report.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    
    # Check if a 'reports' directory exists, if not, create it
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    # --- Generate all 5 charts ---
    plot_accuracy_progression()
    plot_base_model_comparison()
    plot_meta_model_comparison()
    plot_final_confusion_matrix()
    plot_final_classification_report()

    print("\n--- All 5 charts have been generated and saved as .png files! ---")