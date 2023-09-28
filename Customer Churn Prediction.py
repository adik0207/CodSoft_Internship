import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Data Collection and Preparation
data = pd.read_csv('Customer churn dataset/Churn_Modelling.csv')

# Data Cleaning and Feature Engineering
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)  # Drop irrelevant columns

# Encoding categorical features
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

# Splitting Data into Train and Test Sets
X = data.drop('Exited', axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model Training
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))