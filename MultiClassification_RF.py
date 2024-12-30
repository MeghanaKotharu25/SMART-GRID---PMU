import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# Load datasets
df = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/PMU_Data_with_Anomalies and Events/TRAINING_DATA')
test_data_1 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 1/TESTING1')
test_data_2 = pd.read_csv('/Users/lakshmikotaru/Documents/Realistic Labelled PMU Data/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competiton 2024_PMU_DATA/SGSMA_Competition Day_Testdata/Competition_Testing Data Set 2/TESTING2')

# Drop unwanted columns
drop_columns = ['EVENT', 'BUSNO']
X = df.drop(columns=drop_columns, errors='ignore')
y = df['EVENT']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
print(y.value_counts())

# Train the Random Forest Classifier with progress bar
print("Training the Random Forest model...")
with tqdm(total=100) as pbar:
    rf_model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=20, 
        min_samples_split=10, 
        min_samples_leaf=5, 
        random_state=42, 
        class_weight="balanced"
    )
    rf_model.fit(train_X, train_y)
    pbar.update(100)

print("Random Forest model trained successfully!")

# Perform cross-validation with progress bar
print("Performing cross-validation...")
with tqdm(total=100) as pbar:
    scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    pbar.update(100)

print("Cross-validation scores:", scores)

# Define function to test the model on new datasets with a progress bar
def test_random_forest_model(rf_model, test_data, drop_columns, test_filename):
    print(f"Testing on {test_filename}...")
    with tqdm(total=100) as pbar:
        # Ensure the test data has only the relevant columns
        test_X = test_data[X.columns]  # Match the training feature columns
        
        # Make predictions
        predictions = rf_model.predict(test_X)

        # Add predictions to the test data
        test_data['predicted_event'] = predictions

        # Save predictions to a CSV file
        test_data.to_csv(test_filename, index=False)
        pbar.update(100)
    
    print(f"Predictions saved to {test_filename}")
    return test_data

# Test on the first test data
result_data_1 = test_random_forest_model(rf_model, test_data_1, drop_columns, 'predictions_test_1.csv')

# Test on the second test data
result_data_2 = test_random_forest_model(rf_model, test_data_2, drop_columns, 'predictions_test_2.csv')

# Evaluate on split test data from training dataset
print("Evaluating on the split test data...")
with tqdm(total=100) as pbar:
    predictions_test = rf_model.predict(test_X)
    accuracy = accuracy_score(test_y, predictions_test)
    pbar.update(100)

print(f"Evaluation Accuracy on split test data: {accuracy}")
print("Classification Report for split test data:")
print(classification_report(test_y, predictions_test))

# Optionally print some rows from the output files
print("Results for Test Data 1 (first 5 rows):")
print(result_data_1.head())

print("Results for Test Data 2 (first 5 rows):")
print(result_data_2.head())
