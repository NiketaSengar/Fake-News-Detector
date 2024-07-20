import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the dataset
df = pd.read_csv('train.csv')

# Fill missing values with empty strings
df = df.fillna('')

# Preprocess the text column
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Streamlit app
def predict(text):
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return prediction[0]

st.title('Fake News Detection App')

user_input = st.text_area("Enter the news article text:")

if st.button('Classify'):
    if user_input:
        prediction = predict(user_input)
        if prediction == 1:
            st.write("**Prediction:** Fake News")
            st.toast('Fake News !!', icon='⚠️')
        else:
            st.write("**Prediction:** Real News")
            st.toast('Correct News', icon='✔️')
    else:
        st.write("Please enter some text to classify.")
