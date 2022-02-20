import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle

# It's the same pre processing as in Keras

#               PREPROCESSING     START              #


stopWordsEnglish = stopwords.words('english')

punctuation = [".", ";", ":",  "?", "(", ")", "[", "]", "\"",
               "\'", "!", "...", "..", "-", "/", "*", "`", "``", "''", "_"]
stopWordsEnglish.extend(punctuation)

# Read dataset
og_df = pd.read_csv('./dataset/valid.csv', encoding='utf8')

og_df['email'] = og_df['email'].astype(str)

# Tokenize emails and remove stopwords including punctuation which i have just added to the stop word list
og_df['token'] = og_df['email'].apply(word_tokenize)
og_df['without_stopwords'] = og_df['token'].apply(lambda x: [item.lower(
) for item in x if item.lower() not in stopWordsEnglish and not item.isnumeric()])




training = []

processed = pd.DataFrame()

emails = og_df['without_stopwords']
vectorized_emails = []
flags = og_df['flag']
# make flags numeric
flags = flags.replace({'spam': 1, 'ham': 0}, regex=True)

#               PREPROCESSING     OVER              #


#              Vectorization Start            #
model = Doc2Vec(
    min_count=1,
    window=6,
    vector_size=100,
    sample=1e-4,
    negative=5,
    workers=4)


# Train doc2vec to vectorize phrases
for i in range(len(og_df)):
    training.append(TaggedDocument(words=emails[i], tags=[i]))


model.build_vocab(training)

model.train(training,
            total_examples=model.corpus_count,
            epochs=40)

model.save('d2vModel_knn.d2v')
# Vectorize all emails
for i in range(len(emails)):
    vectorized_emails.append(model.infer_vector(emails[i]))

processed['flags'] = flags
processed['emails'] = vectorized_emails


emails_train, emails_test, flags_train, flags_test = train_test_split(
    vectorized_emails, flags, test_size=0.3, random_state=0)

# Create knn classifier
classifier = KNeighborsClassifier(n_neighbors=30)
classifier.fit(emails_train, flags_train)
y_pred = classifier.predict(emails_test)


def check_validity(word):
    if(word.lower() not in stopWordsEnglish and not word.isnumeric()):
        return True
    return False

# Clean message input and return vectorized
def format_input(test):
    tokenized_sentence = word_tokenize(test)
    final = list(filter(check_validity, tokenized_sentence))
    print(final)
    return model.infer_vector(final)

# Return true or false, true for spam and false for ham
def predict_input(input):
    formatted_input = format_input(input)
    # print(formatted_input)
    if classifier.predict([formatted_input])[0]:
        print(True)
        return True
    print(False)
    return False

predict_input('Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066')

# print('accuracy %s' % accuracy_score(flags_test, y_pred))
# print('F1 score: {}'.format(f1_score(flags_test, y_pred, average='weighted')))
# print('recall: {}'.format( recall_score(flags_test, y_pred, average='weighted')))
# print('precision: {}'.format(precision_score(flags_test, y_pred, average='weighted')))

pickle.dump(classifier, open("model_knn.sav", 'wb'))