import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import joblib

# Load your trained meta-model
model = joblib.load('stacking_meta_model_xgb_v4.pkl')
# Load features/labels for calibration set (preferably validation)
X_val = np.load('oof_test_features_v4.npy')
y_val = np.load('oof_test_labels_v4.npy')

# (Re)fit calibrator wrappers on meta-model using validation data
platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
platt.fit(X_val, y_val)
isotonic = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
isotonic.fit(X_val, y_val)

# Get probability predictions
prob_platt = platt.predict_proba(X_val)
prob_iso = isotonic.predict_proba(X_val)

# Scores: lower is better (Brier=prob-calibration, logloss=stricter, ECE needs custom code)
brier_platt = brier_score_loss((y_val==1).astype(int), prob_platt[:,1])
brier_iso = brier_score_loss((y_val==1).astype(int), prob_iso[:,1])
logloss_platt = log_loss(y_val, prob_platt)
logloss_iso = log_loss(y_val, prob_iso)

# Plot Reliability Diagram for each
plt.figure(figsize=(12,5))
for idx, (probs, name) in enumerate(zip([prob_platt, prob_iso], ["Platt (Sigmoid)", "Isotonic"])):
    plt.subplot(1,2,idx+1)
    for i, cname in enumerate(['pre', 'peak', 'post']):
        frac_pos, mean_pred = calibration_curve((y_val==i).astype(int), probs[:,i], n_bins=10)
        plt.plot(mean_pred, frac_pos, marker='o', label=cname.title())
    plt.plot([0,1],[0,1],'k--',label='Perfect')
    plt.title(f"{name} Calibration")
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.legend()
plt.tight_layout()
plt.savefig("calibration_comparison.png", dpi=200)
plt.show()

print(f"Platt Brier score (class 1): {brier_platt:.5f}, Logloss: {logloss_platt:.5f}")
print(f"Isotonic Brier score (class 1): {brier_iso:.5f}, Logloss: {logloss_iso:.5f}")

# Decision: Lower Brier/logloss and better curve => preferred calibration!
