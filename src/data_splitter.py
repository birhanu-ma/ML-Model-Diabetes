from sklearn.model_selection import train_test_split
import pandas as pd
import os

class DataSplitter:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_and_save(self, test_size=0.2, random_state=42, folder_path="../data/Preprocessing/splitted"):
        # 1. Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # 2. Save the splits (we save these so we have a record of the raw split)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        X_train.to_csv(f"{folder_path}/X_train.csv", index=False)
        X_test.to_csv(f"{folder_path}/X_test.csv", index=False)
        y_train.to_csv(f"{folder_path}/y_train.csv", index=False)
        y_test.to_csv(f"{folder_path}/y_test.csv", index=False)

        print(f"✅ Split complete. Raw split files saved to {folder_path}")
        return X_train, X_test, y_train, y_test