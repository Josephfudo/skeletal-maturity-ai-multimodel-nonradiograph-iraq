import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# --- 1. Load Your Data ---
# Load the training data created by your stratified_split.py
# This assumes the CSV has all your new engineered features
try:
    train_df = pd.read_csv('train_metadata.csv')
except FileNotFoundError:
    print("Error: train.csv not found.")
    # Handle error, maybe point to the right file
    exit()

# Define your target column and feature columns
TARGET_COLUMN = 'skeletal_maturity_stage' # Change this to your actual target column
# Get all columns that are not the target or subject ID
FEATURES = [col for col in train_df.columns if col not in [TARGET_COLUMN, 'Case_ID']]

X_train = train_df[FEATURES]
y_train = train_df[TARGET_COLUMN]

print(f"Loaded {len(X_train)} training samples with {len(FEATURES)} features.")

# --- 2. Define the Hyperparameter Search Space ---
# We use distributions (like randint, uniform) for a random search
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3), # Continuous range from 0.01 to 0.31
    'subsample': uniform(0.7, 0.3),      # Continuous range from 0.7 to 1.0
    'colsample_bytree': uniform(0.7, 0.3),# Continuous range from 0.7 to 1.0
    'gamma': uniform(0, 0.5)
}

# --- 3. Set Up the Randomized Search ---
xgb_model = XGBClassifier(
    objective='multi:softmax', # Use 'binary:logistic' for 2 classes
    num_class=y_train.nunique(),  # Assumes multi-class classification
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

# n_iter: How many different combinations to try. 100 is a good start.
# cv: Number of cross-validation folds. 5 is standard.
# n_jobs=-1: Use all available CPU cores to speed up the search.
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1,
    random_state=42 # for reproducible results
)

print("Starting hyperparameter tuning...")

# --- 4. Run the Search ---
# This will take some time!
random_search.fit(X_train, y_train)

# --- 5. Show the Results ---
print("\n--- Tuning Complete ---")
print(f"Best cross-validation accuracy: {random_search.best_score_ * 100:.2f}%")
print("\nBest parameters found:")
print(random_search.best_params_)