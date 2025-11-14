import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Load true test labels and predicted probabilities
y_true = np.load('oof_test_labels_v4.npy')
y_prob = np.load('xgb_stack_test_proba_v4.npy')  # shape: (n_samples, n_classes)

class_names = ['pre', 'peak', 'post']

plt.figure(figsize=(8, 6))
for i, cname in enumerate(class_names):
    prob_pos = y_prob[:, i]
    true_bin = (y_true == i).astype(int)
    frac_pos, mean_pred = calibration_curve(true_bin, prob_pos, n_bins=10, strategy='uniform')
    plt.plot(mean_pred, frac_pos, marker='o', label=f"{cname.title()}")

plt.plot([0,1], [0,1], 'k--', lw=2, label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title("Calibration Plot (Reliability Diagram)\nXGBoost Stacking v4 Final Model")
plt.legend()
plt.tight_layout()
plt.savefig('calibration_plot_xgb_stacking_v4.png', dpi=200)
plt.show()
