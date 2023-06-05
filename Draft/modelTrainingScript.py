# import pandas as pd
# import itertools

# from modelTuning import train_models_with_hyperparameter_tuning

# df = pd.read_csv('preprocessedData.csv')

# X = df.copy().drop(columns=['good_indicator', 'bad_indicator'])
# y = df[['bad_indicator']]


# Define the models you want to train
# models = [
#     "Logistic Regression",
#     "Support Vector Machine",
#     "Decision Tree",
#     "Random Forest",
#     "Gradient Boosting",
#     "XGBoost",
#     "LightGBM",
#     "K-Nearest Neighbors",
#     "Naive Bayes"
# ]

# # Define the hyperparameters for each model
# hyperparameters = [
#     {"C": [1.0, 10.0], "solver": ["liblinear"]},  # Logistic Regression
#     {"C": [1.0, 10.0], "kernel": ["rbf"]},  # Support Vector Machine
#     {"max_depth": [5, 10, None]},  # Decision Tree
#     {"n_estimators": [100, 1000], "max_depth": [5, 10]},  # Random Forest
#     {"n_estimators": [100, 1000], "learning_rate": [0.1, 0.01]},  # Gradient Boosting
#     {"n_estimators": [100, 1000], "learning_rate": [0.1, 0.01]},  # XGBoost
#     {"n_estimators": [100, 1000], "learning_rate": [0.1, 0.01]},  # LightGBM
#     {"n_neighbors": [5, 10]},  # K-Nearest Neighbors
#     {}  # Naive Bayes doesn't have hyperparameters
# ]

# print('check')
# # Generate all combinations of hyperparameters
# hyperparameter_combinations = list(itertools.product(*[param.values() for param in hyperparameters]))
# print(hyperparameter_combinations)

# for params in hyperparameter_combinations:
#     print("Hyperparameters:", params)
#     trained_models, best_model, best_model_name, best_metrics = train_models_with_hyperparameter_tuning(models, [params]*len(models), X, y, n_splits=5)
#     print("--------------------------")