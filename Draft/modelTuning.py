from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd 


def train_models_with_hyperparameter_tuning(models, hyperparameters, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    trained_models = []
    best_model = None
    best_model_name = ""
    best_metrics = {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
    
    results = pd.DataFrame(columns=["Model", "Hyperparameters", "Accuracy", "Precision", "Recall", "F1 Score"])
    
    for model_name, model_params in zip(models, hyperparameters):
        if model_name == "Logistic Regression":
            model = LogisticRegression(**dict(model_params))
        elif model_name == "Support Vector Machine":
            model = SVC(**dict(model_params))
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(**dict(model_params))
        elif model_name == "Random Forest":
            model = RandomForestClassifier(**dict(model_params))
        elif model_name == "Gradient Boosting":
            model = GradientBoostingClassifier(**dict(model_params))
        elif model_name == "XGBoost":
            model = xgb.XGBClassifier(**dict(model_params))
        elif model_name == "LightGBM":
            model = lgb.LGBMClassifier(**dict(model_params))
        elif model_name == "K-Nearest Neighbors":
            model = KNeighborsClassifier(**dict(model_params))
        elif model_name == "Naive Bayes":
            model = GaussianNB()
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            metrics["accuracy"].append(accuracy)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
        
        avg_metrics = {
            "accuracy": sum(metrics["accuracy"]) / n_splits,
            "precision": sum(metrics["precision"]) / n_splits,
            "recall": sum(metrics["recall"]) / n_splits,
            "f1": sum(metrics["f1"]) / n_splits
        }
        
        trained_models.append(model)
        
        results = results.append({
            "Model": model_name,
            "Hyperparameters": model_params,
            "Accuracy": avg_metrics["accuracy"],
            "Precision": avg_metrics["precision"],
            "Recall": avg_metrics["recall"],
            "F1 Score": avg_metrics["f1"]
        }, ignore_index=True)
        
        print(f"Model: {model_name}")
        print(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall: {avg_metrics['recall']:.4f}")
        print(f"Average F1 Score: {avg_metrics['f1']:.4f}")
        print("--------------------------")
        
        if avg_metrics["accuracy"] > best_metrics["accuracy"]:
            best_metrics["accuracy"] = avg_metrics["accuracy"]
            best_metrics["precision"] = avg_metrics["precision"]
            best_metrics["recall"] = avg_metrics["recall"]
            best_metrics["f1"] = avg_metrics["f1"]
            best_model = model
            best_model_name = model_name
    
    print("Best Model:")
    print(f"Model: {best_model_name}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print("--------------------------")
    
    # Save results to CSV
    results.to_csv("hyperparameter_results.csv", index=False)
    
    return trained_models, best_model, best_model_name, best_metrics