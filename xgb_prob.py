import numpy as np
import joblib

X_test = np.load('oof_test_features_v4.npy')
meta_model = joblib.load('stacking_meta_model_xgb_v4.pkl')
probs = meta_model.predict_proba(X_test)  # shape: (n_samples, n_classes)
np.save('xgb_stack_test_proba_v4.npy', probs)
