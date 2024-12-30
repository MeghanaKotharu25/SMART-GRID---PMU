
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# Replace 'file.txt' with your text file name
df = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events/TRAINING_DATA')
rows, columns = df.shape

# Add a header row as data at the top
print(rows,",",columns)


df1 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 1/TESTING1')
print(df1.head())
rows, columns = df1.shape
print(rows,",",columns)


df2 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 2/TESTING2')
print(df2.head())
rows, columns = df2.shape
print(rows,",",columns)

drop=['EVENT','BUSNO']

X = df.drop(columns=drop)
y = df['EVENT']



# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

class_counts = np.bincount(train_y)  # y_train is the true labels for training
total = len(train_y)
class_weights = {i: total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}
xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    max_depth=6,  # Slightly increase depth for better feature splits
    learning_rate=0.05,  # Lower learning rate for better convergence
    n_estimators=200,  # Increase the number of trees
    scale_pos_weight=class_weights  # Apply class weights
)
# Initialize XGBoost Classifier
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=5, learning_rate=0.1, n_estimators=100)

# Fit the model on the training data
xgb_model.fit(train_X, train_y)

# Predict and evaluate the model on the test data
preds = xgb_model.predict(test_X)

# Print classification report
print("Accuracy:", accuracy_score(test_y, preds)*100
      )
print("Classification Report for Direct XGBClassifier:")
print(classification_report(test_y, preds))

# Preprocess the test data by removing the 'Event' column (if present)
X_test_1 = df1.drop(columns=drop, errors='ignore')
X_test_2 = df2.drop(columns=drop, errors='ignore')

# Apply scaling if necessary (matching training data preprocessing)
scaler = StandardScaler()
X_test_1_scaled = scaler.fit_transform(X_test_1)  # Apply same scaler as used during training
X_test_2_scaled = scaler.transform(X_test_2)

# Function to predict events for a test dataset
def predict_events(test_data, model, original_data):
    # Make predictions on the test data
    predictions = model.predict(test_data)
    
    # Ensure predictions are in the correct format and align with original_data index
    predictions = pd.Series(predictions, index=original_data.index)  # Align predictions with the index of the original DataFrame
    
    # Add predictions to the original test data
    original_data['predicted_event'] = predictions
    
    return original_data

# Predict for the first testing dataset
df1_with_predictions = predict_events(X_test_1_scaled, xgb_model, df1)

# Predict for the second testing dataset
df2_with_predictions = predict_events(X_test_2_scaled, xgb_model, df2)

# Save the test datasets with predictions to new CSV files
df1_with_predictions.to_csv('/Users/lakshmikotaru/Documents/output_test_1_with_predictions.csv', index=False)
df2_with_predictions.to_csv('/Users/lakshmikotaru/Documents/output_test_2_with_predictions.csv', index=False)

print("Predictions saved to output files.")

# Generate confusion matrix
cm = confusion_matrix(test_y, preds)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'], 
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()