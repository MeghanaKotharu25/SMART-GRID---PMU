import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import early_stopping, log_evaluation


# Load datasets
data = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events/TRAINING_DATA')
test_data_1 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 1/TESTING1')
test_data_2 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 2/TESTING2')

drop_columns = ['EVENT', 'BUSNO']  # EVENT will be the target
X = data.drop(drop_columns, axis=1)  # Drop both EVENT (target) and BUSNO (not needed)
y = data['EVENT']  # Target variable

# Option 1: Analyze the Class Distribution
# Visualize class distribution in the dataset
sns.countplot(x=y)
plt.title("Class Distribution in Original Dataset")
plt.xlabel("Event Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Check the counts of each class
class_counts = y.value_counts()
print("Class Counts:\n", class_counts)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Visualize class distribution in training and test data
sns.countplot(x=y_train)
plt.title("Class Distribution in Training Data")
plt.xlabel("Event Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

sns.countplot(x=y_test)
plt.title("Class Distribution in Test Data")
plt.xlabel("Event Classes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# Option 3: Handle Class Imbalance with Class Weights
# Compute class weights
class_weights = {label: max(class_counts) / count for label, count in class_counts.items()}
print("Class Weights:\n", class_weights)

# Prepare class weights for LightGBM
class_weight_list = [class_weights[label] for label in y_train]

# Train LightGBM Model
train_data = lgb.Dataset(X_train, label=y_train, weight=class_weight_list)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define LightGBM parameters
params = {
    'objective': 'multiclass',
    'num_class': y.nunique(),  # Number of unique classes
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'max_depth': 10,
    'num_leaves': 50,
    'min_data_in_leaf': 20,
    'class_weight': 'balanced',  # Automatically handle imbalance
    'verbose': -1
}

# Using LightGBM's built-in cross-validation with early stopping
cv_results = lgb.cv(
    params,
    train_data,
    nfold=5,
    num_boost_round=1000,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ],
    stratified=True,
    return_cvbooster=True  # Ensure the boosting results are returned
)

# Check the available keys in cv_results
print(cv_results.keys())  # This will give us the correct column names

# Determine the optimal number of boosting rounds from CV
# In some versions, it might be 'multi_logloss-mean' or 'multi_logloss'
optimal_boost_rounds = np.argmin(cv_results['valid multi_logloss-mean'])  # Use argmin to find the minimum log loss
print(f"Optimal number of boosting rounds: {optimal_boost_rounds}")

# Train final model with optimal boosting rounds
lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=optimal_boost_rounds
)

# Evaluate model
print("\nEvaluating model...")
y_pred = lgb_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy and classification report
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Total Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_counts.index, yticklabels=class_counts.index)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
lgb_model.save_model('lgbm_multiclass_model.txt')
print("Model saved as 'lgbm_multiclass_model.txt'.")

