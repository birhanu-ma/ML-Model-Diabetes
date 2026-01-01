import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

class FeatureTransformer:
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None

    def fit_transform_save(self, X_train, X_test, folder_path="../data/Transformed"):
        # 1. Identify Numeric and Categorical columns
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        # 2. Define the Pipeline logic
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])

        # 3. DIFFERENT LOGIC: 
        # Fit & Transform Train / ONLY Transform Test
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        # 4. Reconstruct Feature Names (for the DataFrame columns)
        cat_encoder = self.preprocessor.named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        self.feature_names = numeric_features + cat_names

        # 5. Convert back to DataFrames
        df_train_final = pd.DataFrame(X_train_transformed, columns=self.feature_names)
        df_test_final = pd.DataFrame(X_test_transformed, columns=self.feature_names)

        # 6. CREATE FOLDER AND SAVE everything
        # This line ensures the directory exists before saving files
        os.makedirs(folder_path, exist_ok=True)
        
        df_train_final.to_csv(f"{folder_path}/X_train_transformed.csv", index=False)
        df_test_final.to_csv(f"{folder_path}/X_test_transformed.csv", index=False)
        
        # Save the transformer object (the 'rules') for future use
        joblib.dump(self.preprocessor, f"{folder_path}/feature_transformer.pkl")

        print(f"✅ Transformation complete. Folder created and files saved to {folder_path}")
        return df_train_final, df_test_final