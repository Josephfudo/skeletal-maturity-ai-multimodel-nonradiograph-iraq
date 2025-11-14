import pandas as pd

"""
This script collects and summarizes the results from all training
strategies and experiments conducted during this project.

The data is manually curated from the various log files.

It generates two outputs:
1. A formatted summary printed to the console.
2. A 'full_experiment_summary.csv' file for use in reports.
"""

# --- ALL EXPERIMENT DATA, MANUALLY CURATED FROM LOGS ---
experiments_data = [
    {
        "id": 1,
        "training_method": "Simple Averaging Ensemble (Baseline)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "Base Model Batch Size": 8,
            "TTA Rounds": 10
        },
        "models": ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16 (x5 Folds each)"],
        "training_time": "~12 hours (Base Models) + ~30 min (Prediction)",
        "results": {
            "Accuracy": 79.17,
            "Precision (Macro)": 0.7816,
            "Recall (Macro)": 0.7812,
            "F1 (Macro)": 0.7801
        },
        "data_type": "Image + Tabular",
        "main_point": "Baseline model from Step 6. All 25 models get an equal 1/25 vote. Showed confusion between 'Pre' and 'Post' peak."
    },
    {
        "id": 2,
        "training_method": "Weighted Averaging Ensemble (Method 1)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "Base Model Batch Size": 8,
            "TTA Rounds": 10
        },
        "models": ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16 (x5 Folds each)"],
        "training_time": "~12 hours (Base Models) + ~30 min (Prediction)",
        "results": {
            "Accuracy": 83.00,
            "Precision (Macro)": 0.8349,
            "Recall (Macro)": 0.8452,
            "F1 (Macro)": 0.8325
        },
        "data_type": "Image + Tabular",
        "main_point": "First improvement. Gave more 'vote' to better-performing backbones (like resnet18). Accuracy jumped ~4%."
    },
    {
        "id": 3,
        "training_method": "Stacking v1 (LR, 3-feat)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "TTA Rounds": 0,
            "Meta-Model": "Logistic Regression"
        },
        "models": ["25 Base Models", "1x LogisticRegression"],
        "training_time": "~12 hours (Base Models) + ~5 min (Feature Gen) + <1 min (Meta-Model Train)",
        "results": {
            "Accuracy": 85.00,
            "Precision (Macro)": 0.8601,
            "Recall (Macro)": 0.8597,
            "F1 (Macro)": 0.8554
        },
        "data_type": "Image + Tabular (as features)",
        "main_point": "First stacking test. Fed 3 features (avg. probabilities) to a meta-model. Beat weighted averaging."
    },
    {
        "id": 4,
        "training_method": "Stacking v2 (XGB, 15-feat)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "TTA Rounds": 0,
            "Meta-Model": "XGBoost (Default Params)"
        },
        "models": ["25 Base Models", "1x XGBoost"],
        "training_time": "~12 hours (Base Models) + ~5 min (Feature Gen) + <1 min (Meta-Model Train)",
        "results": {
            "Accuracy": 89.00,
            "Precision (Macro)": 0.8944,
            "Recall (Macro)": 0.8920,
            "F1 (Macro)": 0.8931
        },
        "data_type": "Image + Tabular (as features)",
        "main_point": "Fed 15 features (all 5 model probs) to XGBoost. XGBoost > LR. Solved most 'Pre/Post' confusion."
    },
    {
        "id": 5,
        "training_method": "Stacking v3 (XGB, 25-feat)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "TTA Rounds": 0,
            "Meta-Model": "XGBoost (Default Params)"
        },
        "models": ["25 Base Models", "1x XGBoost"],
        "training_time": "~12 hours (Base Models) + ~5 min (Feature Gen) + <1 min (Meta-Model Train)",
        "results": {
            "Accuracy": 94.00,
            "Precision (Macro)": 0.9417,
            "Recall (Macro)": 0.9417,
            "F1 (Macro)": 0.9417
        },
        "data_type": "Image + Tabular (as features)",
        "main_point": "Fed 25 features (15 model probs + 10 raw tabular) to XGBoost. Huge accuracy jump. Proved raw tabular data was still valuable."
    },
    {
        "id": 6,
        "training_method": "Stacking v3 - Tuned (XGB, 25-feat)",
        "hyperparameters": {
            "Meta-Model Tuning": "100 Optuna Trials (CV=5)",
            "TTA Rounds": 0
        },
        "models": ["25 Base Models", "1x XGBoost (Tuned)"],
        "training_time": "~12 hours (Base Models) + ~1 hour (Meta-Model Tuning)",
        "results": {
            "Accuracy": 87.00,
            "Precision (Macro)": 0.8777,
            "Recall (Macro)": 0.8724,
            "F1 (Macro)": 0.8745
        },
        "data_type": "Image + Tabular (as features)",
        "main_point": "Tuning the XGBoost meta-model on the 94% features. Resulted in overfitting to the CV set and worse test accuracy."
    },
    {
        "id": 7,
        "training_method": "Method 3 (Differential LR Base Models)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "Base Model Batch Size": 8,
            "Learning Rate": "Differential (Backbone LR * 0.1)"
        },
        "models": ["resnet18", "resnet50", "densenet121", "efficientnet_b0", "vgg16 (x5 Folds each)"],
        "training_time": "~12 hours (Base Models)",
        "results": {
            "Accuracy": 74.0,
            "Precision (Macro)": 0.0,
            "Recall (Macro)": 0.0,
            "F1 (Macro)": 0.0
        },
        "data_type": "Image + Tabular",
        "main_point": "Retrained all 25 base models with a smaller LR for backbones. This produced much stronger, more accurate models (e.g., vgg16: 68% -> 75%)."
    },
    {
        "id": 8,
        "training_method": "Stacking v4 (XGB, 25-feat, Diff-LR Models)",
        "hyperparameters": {
            "Base Model Epochs": "100 (w/ patience 10)",
            "TTA Rounds": 0,
            "Meta-Model": "XGBoost (Default Params)"
        },
        "models": ["25 *NEW* Base Models (from Method 3)", "1x XGBoost"],
        "training_time": "~12 hours (Base Models) + ~5 min (Feature Gen) + <1 min (Meta-Model Train)",
        "results": {
            "Accuracy": 96.00,
            "Precision (Macro)": 0.9610,
            "Recall (Macro)": 0.9610,
            "F1 (Macro)": 0.9610
        },
        "data_type": "Image + Tabular (as features)",
        "main_point": "**FINAL & BEST MODEL.** Used the Stacking v3 (25-feature) method, but with the *superior* base models from Method 3. This fixed all remaining confusion."
    }
]

def generate_console_report(data):
    """Prints a formatted summary of all experiments to the console."""
    print("--- [Full Project Experiment Summary] ---")
    for exp in data:
        print("\n" + "="*80)
        print(f"EXPERIMENT {exp['id']}: {exp['training_method']}")
        print("="*80)
        
        print(f"\n  [Main Point]\n  {exp['main_point']}\n")
        
        print("  [Results (Test Set)]")
        print(f"    - Accuracy:         {exp['results']['Accuracy']:.2f}%")
        print(f"    - F1 (Macro):       {exp['results']['F1 (Macro)']:.4f}")
        print(f"    - Precision (Macro):{exp['results']['Precision (Macro)']:.4f}")
        print(f"    - Recall (Macro):   {exp['results']['Recall (Macro)']:.4f}")
        
        print("\n  [Models & Data]")
        print(f"    - Models Used:      {exp['models']}")
        print(f"    - Data Type:        {exp['data_type']}")

        print("\n  [Config & Time]")
        print(f"    - Hyperparameters:  {exp['hyperparameters']}")
        print(f"    - Est. Training Time: {exp['training_time']}")
    print("\n" + "="*80)

def generate_csv_report(data):
    """Saves a flattened CSV file of all experiment results."""
    flat_data = []
    for exp in data:
        row = {
            "ID": exp["id"],
            "Method": exp["training_method"],
            "Main Point": exp["main_point"],
            "Accuracy": exp["results"]["Accuracy"],
            "F1 (Macro)": exp["results"]["F1 (Macro)"],
            "Precision (Macro)": exp["results"]["Precision (Macro)"],
            "Recall (Macro)": exp["results"]["Recall (Macro)"],
            "Data Type": exp["data_type"],
            "Training Time": exp["training_time"],
            "Models": ", ".join(exp["models"]) if isinstance(exp["models"], list) else exp["models"],
            "Hyperparameters": str(exp["hyperparameters"])
        }
        flat_data.append(row)
    
    try:
        df = pd.DataFrame(flat_data)
        df.to_csv("full_experiment_summary.csv", index=False, sep='|')
        print(f"\n--- Successfully saved 'full_experiment_summary.csv' ---")
    except Exception as e:
        print(f"\n--- Error saving CSV file: {e} ---")
        print("Make sure you have pandas installed: pip install pandas")

if __name__ == "__main__":
    generate_console_report(experiments_data)
    generate_csv_report(experiments_data)