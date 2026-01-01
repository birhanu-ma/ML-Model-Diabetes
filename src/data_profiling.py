import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

class DiabetesProfiling:
    """
    Performs data profiling specifically optimized for 
    the Diabetes dataset (clinical measurements) including Outlier Detection.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target = 'glyhb'

    # 1️⃣ DATA OVERVIEW
    def overview(self):
        print("====== DIABETES DATA OVERVIEW ======")
        print(f"Total Observations: {self.df.shape[0]}")
        print(f"Total Features: {self.df.shape[1]}")
        print("\nData Types Summary:")
        display(self.df.dtypes.value_counts())
        print("\nFirst 3 Rows:")
        display(self.df.head(3))

    # 2️⃣ SUMMARY STATISTICS
    def summary_statistics(self):
        print("\n====== SUMMARY STATISTICS ======")
        clinical_labs = ['chol', 'stab.glu', 'hdl', 'ratio', 'glyhb']
        body_measure = ['age', 'height', 'weight', 'waist', 'hip']

        existing_labs = [c for c in clinical_labs if c in self.df.columns]
        existing_body = [c for c in body_measure if c in self.df.columns]

        if existing_labs:
            print("\n--- Clinical / Lab Results ---")
            display(self.df[existing_labs].describe().transpose())

        if existing_body:
            print("\n--- Physical / Body Measurements ---")
            display(self.df[existing_body].describe().transpose())

        categorical_cols = self.df.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            print("\n--- Categorical Variable Distribution ---")
            for col in categorical_cols:
                print(f"\nValue Counts for {col}:")
                display(self.df[col].value_counts())

    # 3️⃣ MISSING VALUES (COUNT + PERCENTAGE)
    def missing_value_summary(self):
        null_count = self.df.isnull().sum()
        null_percent = (null_count / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Values': null_count,
            'Percentage (%)': null_percent
        })
        missing_summary = (
            missing_summary[missing_summary['Missing Values'] > 0]
            .sort_values(by='Percentage (%)', ascending=False)
        )
        print("\n====== MISSING DATA SUMMARY ======")
        if missing_summary.empty:
            print("No missing values found.")
        else:
            display(missing_summary)

    # 4️⃣ DUPLICATE CHECK
    def duplicate_check(self):
        dup_count = self.df.duplicated().sum()
        print("\n====== DUPLICATE ROW CHECK ======")
        print(f"Duplicate Rows Found: {dup_count}")

    # 5️⃣ OUTLIER DETECTION (IQR METHOD)
    def detect_outliers(self):
        """
        Detects outliers using the Interquartile Range (IQR) method.
        Outliers are values below (Q1 - 1.5*IQR) or above (Q3 + 1.5*IQR).
        """
        print("\n====== OUTLIER DETECTION (IQR METHOD) ======")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # We exclude 'id' from outlier detection if it exists
        cols_to_check = [c for c in numeric_cols if c.lower() != 'id']
        
        outlier_data = []

        for col in cols_to_check:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            outlier_data.append({
                'Feature': col,
                'Outlier Count': len(outliers),
                'Percentage (%)': (len(outliers) / len(self.df)) * 100,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })

        outlier_df = pd.DataFrame(outlier_data).sort_values(by='Outlier Count', ascending=False)
        # Only show features that actually have outliers
        display(outlier_df[outlier_df['Outlier Count'] > 0])
        
        print("> Note: High outliers in 'stab.glu' or 'glyhb' are common in diabetic datasets.")

    # 6️⃣ RUN ALL
    def run_all(self):
        self.overview()
        self.summary_statistics()
        self.missing_value_summary()
        self.duplicate_check()
        self.detect_outliers()
        return self.df