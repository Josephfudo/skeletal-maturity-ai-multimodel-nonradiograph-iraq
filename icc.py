import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

# Load arrays from .npy files
labels = np.load('oof_train_labels_v4.npy')               # shape: (n_samples,)
# You must generate preds using your trained XGBoost stacking model!
# For example:
# preds = meta_model.predict(np.load('oof_train_features_v4.npy'))

# Here, for example purposes, let's pretend we have XGBoost predictions already:
# Replace the following line with your model's predictions
preds = np.load('oof_test_labels_v4.npy')
if preds.ndim > 1 and preds.shape[1] > 1:
    preds = np.argmax(preds, axis=1)

def icc_2way_mixed_single(label_array, pred_array):
    data = np.vstack([label_array, pred_array]).T
    n, k = data.shape
    mean_per_target = np.mean(data, axis=1, keepdims=True)
    mean_per_rater = np.mean(data, axis=0, keepdims=True)
    grand_mean = np.mean(data)
    ms_between_targets = np.sum((mean_per_target - grand_mean) ** 2) * k / (n - 1)
    ms_error = np.sum((data - mean_per_target - mean_per_rater + grand_mean) ** 2) / ((k - 1) * (n - 1))
    icc = (ms_between_targets - ms_error) / (ms_between_targets + (k - 1) * ms_error)
    return icc

# ICC (for numeric or ordinal, not always optimal for discrete multiclass)
icc_val = icc_2way_mixed_single(labels, preds)
print("ICC(2,1):", icc_val)

# Cohen's Kappa (for categorical classification agreement)
kappa_val = cohen_kappa_score(labels, preds)
print("Cohen's Kappa:", kappa_val)

# Accuracy (classification accuracy)
acc_val = accuracy_score(labels, preds)
print("Accuracy:", acc_val)
