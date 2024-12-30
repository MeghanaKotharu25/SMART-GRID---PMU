import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events/TRAINING_DATA')

# Transform EVENT column into binary labels: 0 = Normal, 1 = Others
df['Binary_Label'] = df['EVENT'].apply(lambda x: 0 if x == 0 else 1)

# Prepare features and binary labels
X = df.drop(columns=['EVENT', 'Binary_Label', 'BUSNO'], errors='ignore')  # Drop unnecessary columns
y = df['Binary_Label']

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a binary XGBoost classifier
binary_model = XGBClassifier(eval_metric='logloss', random_state=42)
binary_model.fit(train_X, train_y)

# Evaluate the model on the test set
binary_preds = binary_model.predict(test_X)

print("Binary Classification Accuracy:", accuracy_score(test_y, binary_preds))
print("Classification Report for Anomaly Detection:")
print(classification_report(test_y, binary_preds))

# Confusion matrix
cm = confusion_matrix(test_y, binary_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Others'], yticklabels=['Normal', 'Others'])
plt.title('Confusion Matrix for Anomaly Detection')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Load testing datasets
df1 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 1/TESTING1')
df2 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 2/TESTING2')

# Drop irrelevant columns
X_test_1 = df1.drop(columns=['BUSNO'], errors='ignore')
X_test_2 = df2.drop(columns=['BUSNO'], errors='ignore')

# Predict binary labels for testing datasets
test1_preds = binary_model.predict(X_test_1)
test2_preds = binary_model.predict(X_test_2)

# Add predictions to the test datasets
df1['Anomaly_Label'] = test1_preds
df2['Anomaly_Label'] = test2_preds

# Map predictions to "Normal" and "Others"
df1['Anomaly_Label'] = df1['Anomaly_Label'].map({0: 'Normal', 1: 'Others'})
df2['Anomaly_Label'] = df2['Anomaly_Label'].map({0: 'Normal', 1: 'Others'})

# Save the results
df1.to_csv('/Users/lakshmikotaru/Documents/output_test_1_with_anomaly_labels.csv', index=False)
df2.to_csv('/Users/lakshmikotaru/Documents/output_test_2_with_anomaly_labels.csv', index=False)

print("Predictions for test datasets saved.")
