#importing nessasary Python librarys and tools.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib



#loading the provided dataset into a pandas dataframe
df = pd.read_csv('/Users/matthewrothera/Downloads/BCIT601_starter_service_requests_dataset.csv')
pd.set_option('display.max_rows', None)

df['request_text'] = df['request_text'].str.lower()
#converting text to lowercase.

df['request_text'] = df['request_text'].str.strip()
#stripping leading and trailing spaces

df['request_text'] = df['request_text'].str.ljust(100)
#left justifying the request text as there were lines out of alignment.

#printing the dataset.
print(df.head(80))
df = df.drop_duplicates(subset=["request_text"])
#Dropping any duplicates. The duplicate count is now 0.



###################################################################

#Training the model
X = df["request_text"]
y = df["category"]


#splitting the data into training and test sets
test_data_split = 0.2
test_state = 45

X_train, X_test, y_train, y_test = train_test_split(
 X, y,
    test_size= test_data_split,
    random_state= test_state
)


# Vectorizer
vectorizer = TfidfVectorizer()

# transforming the training data
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transforming the test data 
X_test_tfidf = vectorizer.transform(X_test)


#defining the model to be used and fitting the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)


#print the classification report
print(classification_report(y_test, predictions))



results = pd.DataFrame({
    "input_text": X_test,
    "true_label": y_test,
    "predicted_label": predictions
})

results.head(5).to_csv("model_predictions.csv", index=False)
#saving 5 example predictions to a .csv file called "model_predictions.csv"


#saving the model and vectorizer to disk
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
















