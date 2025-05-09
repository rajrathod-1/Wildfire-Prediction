import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


data = pd.read_csv("data-cleaned.csv")

data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes

features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'


data = data.dropna(subset=features + [target])


X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)


rf_model.fit(X_train_smote, y_train_smote)
gb_model.fit(X_train_smote, y_train_smote)


rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)


voting_model = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('gradient_boosting', gb_model)
], voting='soft')


voting_model.fit(X_train_smote, y_train_smote)


voting_predictions = voting_model.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    print(f"--- {name} ---")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")


evaluate_model("Random Forest", y_test, rf_predictions)


evaluate_model("Gradient Boosting", y_test, gb_predictions)


evaluate_model("Voting Classifier", y_test, voting_predictions)
