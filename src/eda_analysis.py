import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class EdaAnalysis:
    """
    Performs Targeted EDA for the Diabetes Dataset:
    Univariate, Bivariate, Multivariate, Correlation, and Outlier Analysis.
    """

    def __init__(self, df: pd.DataFrame):
        # Drop rows with missing target for clean visualization
        self.df = df.dropna(subset=['glyhb'])
        self.target = 'glyhb'
        sns.set_theme(style="whitegrid")

    # ============================================================
    # 1. UNIVARIATE ANALYSIS (Distributions)
    # ============================================================
    def univariate_analysis(self):
        print("Generating Univariate Histograms...")
        features = ['glyhb', 'chol', 'stab.glu', 'hdl', 'ratio', 'age']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(features):
            if col in self.df.columns:
                sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.show()

    # ============================================================
    # 2. OUTLIER VISUALIZATION (Box Plots)
    # ============================================================
    def outlier_visualization(self):
           """
           Generates standard Box Plots to clearly show Medians, 
           Quartiles, and statistical Outliers.
           """
           print("Generating Outlier Box Plots...")
           features = ['chol', 'stab.glu', 'hdl', 'ratio', 'age', 'weight']
           fig, axes = plt.subplots(2, 3, figsize=(18, 10))
           axes = axes.flatten()
   
           for i, col in enumerate(features):
               if col in self.df.columns:
                   # We use y=col to make them vertical, or x=col for horizontal
                   # Assigning x=col and leaving y empty creates the standard box
                   sns.boxplot(x=self.df[col], ax=axes[i], color='lightcoral', 
                               flierprops={"marker": "x"}, fliersize=5)
                   axes[i].set_title(f'Outliers in {col}')
                   axes[i].set_xlabel(col)
           
           plt.tight_layout()
           plt.show()

    # ============================================================
    # 3. BIVARIATE ANALYSIS (Target vs Features)
    # ============================================================
    def bivariate_analysis(self):
        print("Generating Bivariate Plots (Target vs Features)...")
        features = ['chol', 'stab.glu', 'hdl', 'ratio', 'age']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(features):
            if col in self.df.columns:
                sns.regplot(data=self.df, x=col, y=self.target, ax=axes[i], 
                            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                axes[i].set_title(f'{self.target} vs {col}')
        
        fig.delaxes(axes[5])
        plt.tight_layout()
        plt.show()

    # ============================================================
    # 4. MULTIVARIATE ANALYSIS
    # ============================================================
    def multivariate_analysis(self):
        print("Generating Multivariate Plots...")
        
        # Plot 1: glyhb, chol, and stab.glu
        plt.figure(figsize=(10, 6))
        scatter1 = plt.scatter(self.df['chol'], self.df['stab.glu'], 
                               c=self.df['glyhb'], cmap='viridis', alpha=0.6)
        plt.colorbar(scatter1, label='GlyHb %')
        plt.xlabel('Cholesterol')
        plt.ylabel('Stabilized Glucose')
        plt.title('Relationship: Chol vs Stab.Glu (Colored by GlyHb)')
        plt.show()

    # ============================================================
    # 5. CORRELATION ANALYSIS
    # ============================================================
    def correlation_analysis(self):
        print("Generating Correlation Heatmap...")
        plt.figure(figsize=(12, 8))
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def run_full_eda(self):
        self.univariate_analysis()
        self.outlier_visualization() # Added this to the pipeline
        self.bivariate_analysis()
        self.multivariate_analysis()
        self.correlation_analysis()