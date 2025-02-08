# Credit Card Fraud Detection

## Overview
This project implements a machine learning model to detect fraudulent transactions in credit card data using a **Random Forest Classifier**. The dataset used is a CSV file containing transaction features along with a binary label indicating fraud or non-fraud transactions.

## Dataset
- **Source:** The dataset is assumed to be in a file named `creditcard.csv`.
- **Number of Rows:** 284,807
- **Number of Columns:** 31
- **Features:**
  - The dataset contains 30 anonymized numerical features (`V1` to `V28`), `Time`, and `Amount`.
  - `Class` is the target variable, where:
    - `0`: Non-fraudulent transaction.
    - `1`: Fraudulent transaction.
- **Missing Values:** No missing values in the dataset.
- **Class Distribution:**
  - Non-Fraud (0): 99.83%
  - Fraud (1): 0.17%

## Preprocessing Steps
1. **Handling Missing Data**
   - Dropped rows with `NaN` values in the `Class` column.
2. **Feature Selection**
   - Dropped the `Time` column (as it is not relevant).
   - Standardized the `Amount` column using `StandardScaler`.
3. **Train-Test Split**
   - 80% training, 20% testing, stratified by class distribution.
4. **Handling Class Imbalance**
   - Oversampling of the minority class using `resample()`.

## Model Details
- **Algorithm:** `RandomForestClassifier`
- **Hyperparameters:**
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
  - `bootstrap=True`
  - `random_state=42`

## Evaluation Metrics
- **Classification Report** (Precision, Recall, F1-score)
- **ROC-AUC Score**
- **Confusion Matrix**
- **Precision-Recall Curve**
- **Feature Importance Analysis**

## Results
### Classification Report
![Classification Report](file-1Mn5zsAMvGKvyaAJRmkFKm)

### Confusion Matrix
![Confusion Matrix](file-Q6S5zAAaNPdECqKYaeZP8k)

### Precision-Recall Curve
![Precision-Recall Curve](file-Xe7Z6vxyFWFZa44akkr2PF)

### Feature Importance in Fraud Detection
![Feature Importance](file-DaHdFuCRoVp93mvfC9wXAb)

## Installation & Usage
### Dependencies
Install required Python libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Code
1. Ensure `creditcard.csv` is in the correct directory.
2. Execute the script:
```bash
python fraud_detection.py
```

## Visualization
- **Confusion Matrix**: Displays the model's classification performance.
- **Precision-Recall Curve**: Evaluates the model's precision and recall tradeoff.
- **Feature Importance**: Identifies key features in fraud detection.

## Notes
- The dataset is highly imbalanced; hence, oversampling was used to improve performance.
- Random Forest was chosen for its robustness and ability to handle imbalanced data effectively.

## Future Improvements
- Implement more advanced resampling techniques like SMOTE.
- Experiment with other classifiers (XGBoost, LightGBM, etc.).
- Perform hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
- Deploy the model using Flask/FastAPI for real-time fraud detection.

## License
This project is open-source and can be modified and distributed freely.
