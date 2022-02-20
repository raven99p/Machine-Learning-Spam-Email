import pandas as pd
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import sys

d2v_model = Doc2Vec.load('d2vModel_knn.d2v')
stopWordsEnglish = stopwords.words('english')
punctuation = [".", ";", ":",  "?", "(", ")", "[", "]", "\"",
               "\'", "!", "...", "..", "-", "/", "*", "`", "``", "''", "_"]

stopWordsEnglish.extend(punctuation)

def check_validity(word):
    if(word.lower() not in stopWordsEnglish and not word.isnumeric()):
        return True
    return False


def format_input(test):
    tokenized_sentence = word_tokenize(test)
    final = list(filter(check_validity, tokenized_sentence))
    # print(final)
    return d2v_model.infer_vector(final)

model_filename = "model_knn.sav"
knn_model = pickle.load(open(model_filename, 'rb'))

def predict_input(input):
    formatted_input = format_input(input)
    # print(formatted_input)
    if knn_model.predict([formatted_input])[0]:
        print(True)
        return True
    print(False)
    return False

predict_input('Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066')

sys.modules[__name__] = predict_input
