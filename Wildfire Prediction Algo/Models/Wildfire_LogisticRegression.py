import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

start_time = time.time()

data = pd.read_csv("data-cleaned.csv")

categorical = ['fire_position_on_slope']
numerical = ['wind_speed', 'relative_humidity', 'temperature', 'fire_spread_rate', 'current_size', 'assessment_hectares']

for var in categorical:
    data[var] = data[var].fillna("N/A")
    print(f"{var} value counts after filling missing values:")
    print(data[var].value_counts())

for var in numerical:
    data[var] = data[var].fillna(data[var].median())

print("\nNull values after cleanup:")
print(data.isnull().sum())

data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes

features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
}

grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)

best_params = grid_search.best_params_
print("Best Hyperparameters from GridSearchCV:", best_params)

print("\nGridSearchCV Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results)

best_model = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=500)

fit_start_time = time.time()
best_model.fit(X_train_smote, y_train_smote)
fit_end_time = time.time()

fit_time = fit_end_time - fit_start_time
print(f"\nModel Fitting Time: {fit_time:.2f} seconds")

y_pred = best_model.predict(X_test)

report_dict = classification_report(y_test, y_pred, output_dict=True)
precision_weighted_avg = report_dict['weighted avg']['precision']
print(f"Precision for the whole model: {precision_weighted_avg}")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Accuracy Score after Tuning: {accuracy}")

print("Best Classification Report:")
print(classification_report(y_test, y_pred))

coefficients = best_model.coef_
print(f"Best Model Coefficients: {coefficients}")

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
