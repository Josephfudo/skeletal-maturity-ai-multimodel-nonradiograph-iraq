import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

"""
This script generates a single, large .png file containing a 3x2 grid
of confusion matrices for all major models developed in this project.

It allows for a direct visual comparison of how performance improved
and how specific errors were eliminated at each stage.
"""

def plot_all_confusion_matrices():
    print("Generating 'all_models_confusion_matrix.png'...")
    
    # --- Data from all your log files ---
    # This dictionary holds the title and the confusion matrix data for each model.
    # I've selected the 6 most important models that show your progress.
    cm_data = {
        'A: Simple Avg (79.17%)': np.array([[33, 1, 5], [2, 28, 0], [4, 0, 27]]),
        'B: Weighted Avg (83.00%)': np.array([[26, 4, 9], [2, 28, 0], [2, 0, 29]]),
        'C: Stack v1 (LR, 3-feat) (85.00%)': np.array([[29, 1, 9], [3, 27, 0], [2, 0, 29]]),
        'D: Stack v2 (XGB, 15-feat) (89.00%)': np.array([[34, 2, 3], [2, 28, 0], [4, 0, 27]]),
        'E: Stack v3 (XGB, 25-feat) (94.00%)': np.array([[36, 1, 2], [1, 29, 0], [2, 0, 29]]),
        'F: Stack v4 (XGB, 25-feat, Diff-LR) (96.00%)': np.array([[37, 1, 1], [1, 29, 0], [1, 0, 30]])
    }
    
    labels = ['Pre-peak', 'Peak', 'Post-peak']

    # --- Create the 3x2 grid ---
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 24))
    
    # Flatten the 3x2 grid into a 1D array for easy iteration
    axes = axes.flatten()

    # --- Plot each confusion matrix ---
    for i, (title, cm) in enumerate(cm_data.items()):
        ax = axes[i]
        
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    annot_kws={"size": 16, "weight": "bold"})
        
        ax.set_title(title, fontsize=16, pad=20, weight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, rotation=0, fontsize=10)

    # Add a main title for the entire figure
    fig.suptitle('Progression of Confusion Matrices by Model Strategy', fontsize=24, y=1.03, weight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the final combined chart to the 'reports' directory
    output_filename = 'reports/all_models_confusion_matrix.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"--- Successfully saved comparison chart to '{output_filename}' ---")

if __name__ == "__main__":
    
    # Check if a 'reports' directory exists, if not, create it
    if not os.path.exists('reports'):
        os.makedirs('reports')
        
    plot_all_confusion_matrices()