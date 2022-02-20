import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

# Get stopwords
stopWordsEnglish = stopwords.words('english')

punctuation = [".", ";", ":",  "?", "(", ")", "[", "]", "\"",
               "\'", "!", "...", "..", "-", "/", "*", "`", "``", "''", "_", ">", ">>", "<", "<<", "&", "|", "||", "&&", "=", "-", "+", "@", "\\", "#", "$", "%", "^"]
# Add punctuation to stopwords
stopWordsEnglish.extend(punctuation)

# Read dataset
og_df = pd.read_csv('./dataset/valid.csv', encoding='utf8')

og_df['email'] = og_df['email'].astype(str)

# Tokenize, remove numbers, stopwords, puncuation and lower the words
og_df['token'] = og_df['email'].apply(word_tokenize)
og_df['without_stopwords'] = og_df['token'].apply(lambda x: [item.lower(
) for item in x if item.lower() not in stopWordsEnglish and not item.isnumeric()])







training = []

emails = og_df['without_stopwords']
vectorized_emails = []
flags = og_df['flag']
# make flags numeric
flags = flags.replace({'spam': 1, 'ham': 0}, regex=True)

#               PREPROCESSING     OVER              #

#              Vectorization Start            #
model = Doc2Vec(
    min_count=1, # minimum number of times a word has to appear to not be ingored
    window=7, # windows of bag of words
    vector_size=100, # final vector size
    sample=1e-4,
    negative=5,
    workers=4)# doesn't matter its for multiprocessing


# Train doc2vec to vectorize phrases
for i in range(len(og_df)):
    training.append(TaggedDocument(words=emails[i], tags=[i]))

# Build vocabulary from all the words in the emails
model.build_vocab(training)

# Train for 40 epochs
model.train(training,
            total_examples=model.corpus_count,
            epochs=40)

# Save doc2vec model 
model.save('d2vModel_keras.d2v')

# Vectorize all emails
for i in range(len(emails)):
    vectorized_emails.append(model.infer_vector(emails[i]))

# Split Data
emails_train, emails_test, flags_train, flags_test = train_test_split(
    vectorized_emails, flags, test_size=0.3, random_state=0)

# Metrics
def recall_metric(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positives + K.epsilon())
    return precision


def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Create model.
model = Sequential()
model.add(Dense(14, input_dim=100, activation='relu'))  # input layer
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # output layer

model.compile(loss='binary_crossentropy',  # for binary classification
              optimizer='adam',
              metrics=['accuracy'] # didnt add metrics because it causes problems
              )                    # to the predict function

model.fit(np.array(emails_train),
          np.array(flags_train),
          epochs=20,
          batch_size=20,
          validation_data=(np.array(emails_test), np.array(flags_test)))
model.save('model_keras')
