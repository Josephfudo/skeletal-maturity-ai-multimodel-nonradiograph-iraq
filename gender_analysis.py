import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Data prep (replace with your actual loading if needed)
df = pd.read_csv('test_metadata.csv')
y_true = np.load('oof_test_labels_v4.npy')
y_pred = np.load('xgb_stack_test_preds_v4.npy')
df['y_true'] = y_true
df['y_pred'] = y_pred

metrics = ["Accuracy", "F1 (Macro)", "Precision (Macro)", "Recall (Macro)"]
def compute_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro")
    ]

girls = df['Gender_enc_eng'] == 1
boys  = df['Gender_enc_eng'] == 0

results = {
    "Metric": metrics,
    "Girls": compute_metrics(df.loc[girls, 'y_true'], df.loc[girls, 'y_pred']),
    "Boys":  compute_metrics(df.loc[boys,  'y_true'], df.loc[boys,  'y_pred']),
}
results = pd.DataFrame(results)

color_girls = "#D55E00"   # Academic orange
color_boys = "#0072B2"    # Academic blue

# --- 1. Improved Grouped Bar Chart with External Legend ---
import matplotlib.patches as mpatches
plt.figure(figsize=(10,6))
bar_width = 0.35
x = np.arange(len(metrics))
plt.bar(x - bar_width/2, results["Girls"], bar_width, color=color_girls, label="Girls")
plt.bar(x + bar_width/2, results["Boys"],  bar_width, color=color_boys, label="Boys")
for i in range(len(metrics)):
    plt.text(x[i] - bar_width/2, results['Girls'][i]+0.03, f"{results['Girls'][i]*100:.0f}%", 
             ha='center', va='bottom', fontsize=10, color=color_girls, fontweight='bold')
    plt.text(x[i] + bar_width/2, results['Boys'][i]+0.03, f"{results['Boys'][i]*100:.0f}%",
             ha='center', va='bottom', fontsize=10, color=color_boys, fontweight='bold')
plt.xticks(x, metrics, fontsize=10)
plt.ylim(0, 1.12)
plt.ylabel("Score", fontsize=12)
plt.title("Test Performance by Metric and Gender", fontsize=15)
# Use academic rectangular legend outside the main plot
patches = [mpatches.Patch(color=color_girls, label='Girls'), mpatches.Patch(color=color_boys, label='Boys')]
plt.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.savefig('gender_metrics_grouped_barrrrr.png', dpi=200)
plt.show()
