# Importing necessary libraries required
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading dataset
data = pd.read_csv('C:/Users/hp/Downloads/archive/creditcard.csv')

# Splitting the dataset into features (x) and the target variable (y)
x = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

# Create and train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=50, random_state=40)
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_report_str)
