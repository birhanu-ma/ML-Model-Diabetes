import pandas as pd
import numpy as np
import os

class DataPreprocessing:
    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        # 1. Drop columns with > 50% missing data and ID
        cols_to_drop = ['id', 'bp.2s', 'bp.2d']
        self.df = self.df.drop(columns=[c for c in cols_to_drop if c in self.df.columns])

        # 2. Drop rows where TARGET is missing
        self.df = self.df.dropna(subset=['glyhb'])

        # 3. Impute remaining numeric features with Median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # 4. Impute categorical features with Mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    def handle_outliers_iqr(self, columns):
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df[col] = self.df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
        return self.df

    def separate_features_target(self, target_col='glyhb'):
        """Separates the data into X and y variables."""
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return X, y

    def save_data(self, X, y, folder_path="../data/processed"):
        """Saves the cleaned features and target to disk."""
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        X.to_csv(f"{folder_path}/X_cleaned.csv", index=False)
        y.to_csv(f"{folder_path}/y_cleaned.csv", index=False)
        print(f"✅ Cleaned data saved successfully to {folder_path}")