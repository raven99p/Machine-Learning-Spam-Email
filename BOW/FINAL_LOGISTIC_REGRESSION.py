import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pickle
# Read column as list literal and not list inside list.
vectorized_emails = []
vectorizer = CountVectorizer()
og_df = pd.read_csv('./dataset/valid.csv', encoding='utf8')


vectorizer.fit(og_df['email'].tolist())
emails = og_df['email'].astype(str)
flags = og_df['flag']
flags = flags.replace({'spam': 1, 'ham': 0}, regex=True)

# Vectorize all emails
for i in range(len(emails)):
    vector = vectorizer.transform([emails[i]])
    vectorized_emails.append(vector.toarray()[0])

# Split data.
emails_train, emails_test, flags_train, flags_test = train_test_split(
    vectorized_emails, flags, test_size=0.3, random_state=0)

# # Create classifier train and predict.
logisticRegr = LogisticRegression()
logisticRegr.fit(emails_train, flags_train)
y_pred = logisticRegr.predict(emails_test)


def predict_input(input):
    formatted_input = vectorizer.transform([input]).toarray()
    if logisticRegr.predict([formatted_input[0]])[0]:
        print('true')
        return True
    print('false')
    return False


print('accuracy %s' % accuracy_score(flags_test, y_pred))

ham = 'U in town alone?'
spam = 'Free msg: Single? Find a partner in your area! 1000s of real people are waiting to chat now!Send CHAT to 62220Cncl send STOPCS 08717890890å£1.50 per msg'
predict_input(ham)

pickle.dump(logisticRegr, open("./saved_models/model_logistic_regression.sav", 'wb'))

pickle.dump(vectorizer, open("./saved_models/vectorizer.pkl", 'wb'))