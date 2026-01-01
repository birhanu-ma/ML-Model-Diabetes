# 🩺 Diabetes Risk Factor Analysis & GlyHb Prediction

This project implements a **complete machine learning pipeline** to analyze risk factors associated with Diabetes using the **hbiostat dataset**.

Two primary problems are addressed:

* **Regression:** Predicting the exact concentration of **Glycosylated Hemoglobin (GlyHb)**
* **Classification:** Diagnosing whether a patient is diabetic based on clinical thresholds
  *(A1c ≥ 6.5)*

---

## 📂 Project Structure

```plaintext
├── data/
│   ├── Raw/                # Original hbiostat dataset
│   ├── Preprocessing/      # Cleaned data and split sets (y_train, y_test)
│   └── Transformed/        # Scaled and encoded features (X_train_final, etc.)
│
├── notebooks/
│   ├── eda_analysis.ipynb        # Initial data discovery
│   ├── data_Preprocessing.ipynb  
|   ├── eda_analysis.ipynb  
|   ├── data_profiling.ipynb 
|   ├── data_splitter.ipynb  # Cleaning and imputation
│   ├── feature_Transformer.ipynb  # Scaling and encoding
│   └── model_training.ipynb  
├── src/     
│   ├── eda_analysis.py       
│   ├── data_Preprocessing.py  
|   ├── eda_analysis.py  
|   ├── data_profiling.py 
|   ├── data_splitter.py  
│   ├── feature_Transformer.py
│   └── model_training.py 
│
├── README.md   
├── .gitignore                # Project documentation
└── requirements.txt        # Project dependencies
```

---

## 🚀 Execution Order (Priority)

To reproduce the results, notebooks **must be executed in the following order**.
Each notebook generates the required outputs for the next step.

### **1️⃣ 01_EDA_Profiling.ipynb**

**Goal:** Perform Exploratory Data Analysis (EDA) to understand feature distributions and correlations.

### **2️⃣ 02_Data_Preprocessing.ipynb**

**Goal:** Handle missing values and perform the train–test split.

### **3️⃣ 03_Feature_Engineering.ipynb**

**Goal:** Apply feature scaling and categorical encoding to the split data.

### **4️⃣ 04_Model_Training.ipynb**

**Goal:** Run the `ModelTrainer` for both regression and classification tasks.

---

## 🛠️ Key Features: `ModelTrainer` Class

The project utilizes a custom-built **`ModelTrainer`** class that automates:

* **Multi-Algorithm Evaluation**
  Simultaneous comparison of 12 models, including **XGBoost**, **CatBoost**, **Random Forest**, and **SVM**.

* **Dual-Task Compatibility**
  Seamless handling of both:

  * Continuous targets (Regression)
  * Categorical targets (Classification)

* **Feature Importance Analysis**
  Automatic generation of visualizations to identify key clinical risk factors
  (Research Question 2 – RQ2).

---

## 📈 Summary of Findings

### 🔍 Q1: Regression Results

* **Top Performer:**
  🥇 **CatBoost** achieved the highest (R^2) score and the lowest RMSE.

* **Key Risk Factors:**

  * Patient Age
  * Cholesterol
  * BMI

  These features were the strongest predictors of **GlyHb concentration**.

---

### 🧪 Q2: Classification Results

* **Best Models:**
  **Random Forest**, **XGBoost**, and **KNN** tied for top performance with **88.46% accuracy**.

* **Clinical Insight:**
  Despite high accuracy, the model achieved a **Recall of 0.50**, meaning it correctly identifies only **50% of true diabetic cases**.
  This highlights the need for **sensitivity-focused tuning** in medical applications.

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/birhanu-ma/ML-Model-Diabetes.git
cd ML-Model-Diabetes
```

### 2️⃣ Install Dependencies

This project requires **specific library versions** to ensure reproducibility,
particularly **scikit-learn 1.3.2**.

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the Project

Open Jupyter Notebook or Jupyter Lab and start with:

```text
01_EDA_Profiling.ipynb
```

---

✅ **Designed for reproducibility, clinical interpretability, and extensibility**
