import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Loading and Prepare Data
# Load the dataset from the CSV file
df = pd.read_csv('D:/Engineering/CodSoft/Movie Genre Classification/Movie Genre Classification 1/wiki_movie_plots_deduped.csv')

X = df['Plot']
y = df['Genre']

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

# Applying preprocessing to the text data
X = X.apply(preprocess_text)

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training the Classifier (Multinomial Naive Bayes)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = nb_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

classification_rep = classification_report(y_test, y_pred, zero_division=1)
print("Classification Report:\n", classification_rep)

