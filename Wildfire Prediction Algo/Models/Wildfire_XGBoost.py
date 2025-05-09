import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import time

start_time = time.time()

# Load dataset
data = pd.read_csv("data-cleaned.csv")

# Define categorical and numerical features
categorical = ['fire_position_on_slope']
numerical = ['wind_speed', 'relative_humidity', 'temperature', 'fire_spread_rate', 'current_size', 'assessment_hectares']

# Handle missing values for categorical features
categorical_start_time = time.time()
for var in categorical:
    data[var] = data[var].fillna("N/A")
    print(f"{var} value counts after filling missing values:")
    print(data[var].value_counts())
categorical_end_time = time.time()
categorical_fill_time = categorical_end_time - categorical_start_time
print(f"\nTime to fill categorical missing values: {categorical_fill_time:.2f} seconds")

# Handle missing values for numerical features
numerical_start_time = time.time()
for var in numerical:
    data[var] = data[var].fillna(data[var].median())
numerical_end_time = time.time()
numerical_fill_time = numerical_end_time - numerical_start_time
print(f"\nTime to fill numerical missing values: {numerical_fill_time:.2f} seconds")

# Check for remaining null values
print("\nNull values after cleanup:")
print(data.isnull().sum())

# Convert categorical variable to numeric
data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes

# Define features and target
features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'

# Drop rows with missing values in features or target
data = data.dropna(subset=features + [target])

x = data[features]
y = data[target]

# Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(x_scaled, y)

# Split the data into training and testing sets
X_train_smote, X_test, y_train_smote, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'learning_rate': [0.01, 0.1, 0.2],  # Step size for updating the weights
    'max_depth': [3, 6, 10],  # Maximum depth of the tree
    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight
    'subsample': [0.8, 1.0],  # Fraction of samples to use for fitting each tree
    'colsample_bytree': [0.8, 1.0]  # Fraction of features to use for fitting each tree
}

# Initialize GridSearchCV with XGBClassifier
grid_search = GridSearchCV(estimator=XGBClassifier(eval_metric='logloss', random_state=42),
                           param_grid=param_grid,
                           cv=5,  # Number of folds for cross-validation
                           n_jobs=-1,  # Use all available CPUs
                           verbose=1,  # Print progress
                           scoring='accuracy')  # Use accuracy as the evaluation metric

# Fit the GridSearchCV
grid_search_start_time = time.time()
grid_search.fit(X_train_smote, y_train_smote)
grid_search_end_time = time.time()

grid_search_time = grid_search_end_time - grid_search_start_time
print(f"\nGrid Search CV Time: {grid_search_time:.2f} seconds")

# Best hyperparameters from GridSearchCV
best_params = grid_search.best_params_
print(f"\nBest Hyperparameters from GridSearchCV: {best_params}")

# Get the best model after grid search
best_model = grid_search.best_estimator_

# Predict using the best model
y_pred = best_model.predict(X_test)

# Confusion Matrix and Evaluation Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print("Confusion Matrix:")
print(conf_matrix)
print(f"\nRaw Rates:\n - True Positives (TP): {tp}\n - True Negatives (TN): {tn}\n - False Positives (FP): {fp}\n - False Negatives (FN): {fn}\n")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")

# Total Execution Time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
