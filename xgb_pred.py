import numpy as np
import joblib

# 1. Load the test features for stacking
test_features = np.load('oof_test_features_v4.npy')

# 2. Load your XGBoost stacking model
meta_model = joblib.load('stacking_meta_model_xgb_v4.pkl')

# 3. Generate per-case predictions (label index per sample)
test_preds = meta_model.predict(test_features)  # for sklearn/XGBClassifier .predict returns hard class labels

np.save('xgb_stack_test_preds_v4.npy', test_preds)
print("Saved XGBoost stacking test predictions to xgb_stack_test_preds_v4.npy")
