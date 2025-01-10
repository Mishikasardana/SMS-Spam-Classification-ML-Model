import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('/content/spam.csv', encoding='latin-1')  # Assuming 'latin-1' encoding as before

# Check the column names in your DataFrame
print(df.columns)  # Print the available column names

# Assuming the message column is named 'v2' and the label column is named 'v1' 
# (adjust as needed based on the output above)
message_column_name = 'v2'  # Change to the actual message column name
label_column_name = 'v1' # Change to the actual label column name


# Preprocess data
stop_words = set(stopwords.words('english'))
df[message_column_name] = df[message_column_name].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df[message_column_name])  # Use the correct column name
y = df[label_column_name]  # Use the correct column name for target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred_nb, pos_label='spam'))
print("F1 Score:", f1_score(y_test, y_pred_nb, pos_label='spam'))

# Train and evaluate Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
print("\nLogistic Regression Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred_lr, pos_label='spam'))
print("F1 Score:", f1_score(y_test, y_pred_lr, pos_label='spam'))

# Train and evaluate Support Vector Machine classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)
print("\nSupport Vector Machine Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm, pos_label='spam'))
print("Recall:", recall_score(y_test, y_pred_svm, pos_label='spam'))
print("F1 Score:", f1_score(y_test, y_pred_svm, pos_label='spam'))
