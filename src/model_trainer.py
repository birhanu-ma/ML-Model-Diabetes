import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             accuracy_score, f1_score, precision_score, recall_score)
import joblib
import os

# Import all requested models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, 
                              AdaBoostRegressor, AdaBoostClassifier, 
                              GradientBoostingRegressor, GradientBoostingClassifier)
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import SVR, SVC

class ModelTrainer:
    def __init__(self, task_type='regression'):
        """
        task_type: 'regression' or 'classification'
        """
        self.task_type = task_type
        self.results = []
        self.models = self._initialize_models()

    def _initialize_models(self):
        if self.task_type == 'regression':
            return {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(alpha=0.1),
                "Ridge": Ridge(alpha=1.0),
                "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
                "KNN": KNeighborsRegressor(n_neighbors=5),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
                "CatBoost": CatBoostRegressor(verbose=0, n_estimators=100),
                "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "SVR": SVR(kernel='rbf')
            }
        else: # classification
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "CatBoost": CatBoostClassifier(verbose=0, n_estimators=100),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42)
            }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        self.results = [] # Clear previous results
        for name, model in self.models.items():
            # 1. Fit
            model.fit(X_train, y_train.values.ravel())
            
            # 2. Predict
            preds = model.predict(X_test)
            
            # 3. Score
            if self.task_type == 'regression':
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                mae = mean_absolute_error(y_test, preds)
                self.results.append({"Model": name, "R2 Score": r2, "RMSE": rmse, "MAE": mae})
            else:
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                self.results.append({"Model": name, "Accuracy": acc, "F1 Score": f1, "Recall": rec})
            
            print(f"✅ {name} processing complete.")

        # Return sorted results
        sort_col = "R2 Score" if self.task_type == 'regression' else "F1 Score"
        return pd.DataFrame(self.results).sort_values(by=sort_col, ascending=False)

    def plot_importance(self, model_name, feature_names):
        """Answers RQ2: Key Factors"""
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.Series(importance, index=feature_names).sort_values(ascending=True)
            plt.figure(figsize=(10, 6))
            feature_imp.tail(10).plot(kind='barh')
            plt.title(f"Top 10 Key Factors: {model_name}")
            plt.show()
        else:
            print(f"Model {model_name} does not support feature importance visualization.")