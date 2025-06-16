import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_excel('Raw_Data_v0.xlsx', engine='openpyxl')
df = df.drop(columns=[
    'Ref#', 'Heat treatment', 'Other RM/Rivet/part cost (€/Part)',
    'Gross Weight (g)', 'Other assembled RM/Rivet/part','Heat Treatment cost (€/Part)'
])

num_cols = [
    'Annual target quantity', 'Raw Material Cost (€/kg)', 'Thickness (mm)',
    'Part Net Weight (g)', 'Surface Treatment cost (€/Part)',
    'Final Raw Material cost (€/Part)'
]
cat_cols = [
    'Production', 'Raw Material Designation',
    'Surface Treatment', 'Raw Material'
]

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols] = df[cat_cols].fillna('Missing')
TARGET = 'Total cost with amortization (€/part)'

# Apply square root transformations
for col in num_cols + [TARGET]:
    df[col] = np.sqrt(df[col])

X = df[num_cols + cat_cols]
y = df[TARGET]
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

cost_components = ['Surface Treatment cost (€/Part)',
                   'Final Raw Material cost (€/Part)',]
monotonic_constraints = [
    1 if col in cost_components else 0
    for col in X.columns
]

# Fixed parameters
best_params = {
    'iterations': 1016,
    'learning_rate': 0.024744862272094555,
    'depth': 5,
    'l2_leaf_reg': 1.7300015701528153,
    'border_count': 175,
    'min_data_in_leaf': 16,
    'random_strength': 1.2983378779917965,
    'subsample': 0.9070238688391844,
    'bootstrap_type': 'Bernoulli',
    'feature_border_type': 'MaxLogSum',
    'grow_policy': 'SymmetricTree',
    'monotone_constraints': monotonic_constraints,
    'eval_metric': 'MAPE',
    'verbose': False
}

# Extended validation with 10-fold CV
kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
metrics = {
    'MAE': [], 'RMSE': [], 'MAPE': [], 'R2': [],
    'violations_before': [], 'violations_after': [],
    'within_10%': []
}

for fold, (train_idx, test_idx) in enumerate(kf10.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = CatBoostRegressor(**best_params, cat_features=cat_indices)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    y_pred = model.predict(X_test)
    y_pred_orig = np.square(y_pred)
    y_test_orig = np.square(y_test)

    sum_costs = np.sum(np.square(X_test[cost_components]), axis=1)

    violations_before = np.sum(y_pred_orig < sum_costs)
    y_pred_clipped = np.maximum(y_pred_orig, sum_costs)
    violations_after = np.sum(y_pred_clipped < sum_costs)

    metrics['MAE'].append(mean_absolute_error(y_test_orig, y_pred_clipped))
    metrics['RMSE'].append(np.sqrt(mean_squared_error(y_test_orig, y_pred_clipped)))
    metrics['MAPE'].append(np.mean(np.abs((y_test_orig - y_pred_clipped)/y_test_orig)) * 100)
    metrics['R2'].append(r2_score(y_test_orig, y_pred_clipped))
    metrics['violations_before'].append(violations_before)
    metrics['violations_after'].append(violations_after)
    metrics['within_10%'].append(np.sum(np.abs((y_pred_clipped - y_test_orig)/y_test_orig) <= 0.1))

    # Plot learning curve for last fold
    if fold == 9:
        plt.figure(figsize=(10, 6))
        eval_results = model.get_evals_result()
        plt.plot(eval_results['learn']['MAE'], label='Train MAE')
        if 'validation' in eval_results:
            plt.plot(eval_results['validation']['MAE'], label='Validation MAE')
        plt.title('Learning Curve - MAE Progression')
        plt.xlabel('Iterations')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

# Aggregate results
print("\n=== Final Validation Results ===")
print(f"MAE: {np.mean(metrics['MAE']):.2f} ± {np.std(metrics['MAE']):.2f}")
print(f"RMSE: {np.mean(metrics['RMSE']):.2f} ± {np.std(metrics['RMSE']):.2f}")
print(f"MAPE: {np.mean(metrics['MAPE']):.2f}% ± {np.std(metrics['MAPE']):.2f}")
print(f"R²: {np.mean(metrics['R2']):.2f} ± {np.std(metrics['R2']):.2f}")
print(f"Violations before clipping: {sum(metrics['violations_before'])} ({sum(metrics['violations_before'])/len(X)*100:.2f}%)")
print(f"Violations after clipping: {sum(metrics['violations_after'])} ({sum(metrics['violations_after'])/len(X)*100:.2f}%)")
print(f"Predictions within ±10%: {sum(metrics['within_10%'])/len(X)*100:.2f}%")