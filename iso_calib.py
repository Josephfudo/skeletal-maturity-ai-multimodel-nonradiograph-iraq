import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

# 1. Load true labels and isotonic-calibrated probabilities
y_true = np.load('oof_test_labels_v4.npy')
probs_iso = np.load('xgb_stack_test_proba_isotonic_v4.npy')
class_names = ['pre', 'peak', 'post']

# 2. Calibration Plot (Reliability Diagram) – closely matches your example
plt.figure(figsize=(8, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, cname in enumerate(class_names):
    true_bin = (y_true == i).astype(int)
    frac_pos, mean_pred = calibration_curve(true_bin, probs_iso[:, i], n_bins=10)
    plt.plot(mean_pred, frac_pos, marker='o', label=f"{cname.title()}", color=colors[i])
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfectly calibrated')
plt.ylabel("Fraction of positives", fontsize=12)
plt.xlabel("Mean predicted probability", fontsize=12)
plt.title("Calibration Plot (Reliability Diagram)\nXGBoost Stacking v4 Final Model", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig("calibration_plot_xgb_stacking_v4_isotonic.png", dpi=200)
plt.show()

# 3. Brier Score (per class) and overall Log Loss
print("--- Brier Score (one-vs-rest, all classes) ---")
for i, cname in enumerate(class_names):
    score = brier_score_loss((y_true == i).astype(int), probs_iso[:, i])
    print(f"{cname.title()} Brier score: {score:.5f}")

ll = log_loss(y_true, probs_iso)
print(f"\nMulticlass Log Loss: {ll:.5f}")

# 4. Expected Calibration Error (ECE)
def multiclass_ece(y_true, y_prob, num_bins=10):
    n = len(y_true)
    probs = np.max(y_prob, axis=1)
    preds = np.argmax(y_prob, axis=1)
    bins = np.linspace(0.0, 1.0, num_bins+1)
    ece = 0
    for i in range(num_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if np.any(mask):
            acc = np.mean(y_true[mask] == preds[mask])
            conf = np.mean(probs[mask])
            ece += np.sum(mask) * abs(acc - conf)
    return ece / n

ece_score = multiclass_ece(y_true, probs_iso, num_bins=10)
print(f"\nExpected Calibration Error (ECE): {ece_score:.5f}")
