import pandas as pd
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import sys
import numpy as np
from tensorflow import keras
d2v_model = Doc2Vec.load('d2vModel_keras.d2v')
stopWordsEnglish = stopwords.words('english')

punctuation = [".", ";", ":",  "?", "(", ")", "[", "]", "\"", 'å', 'Ã¥Â',
               "\'", "!", "...", "..", "-", "/", "*", "`", "``", "''", "_", ">", ">>", "<", "<<", "&", "|", "||", "&&", "=", "-", "+", "@", "\\", "#", "$", "%", "^"]

stopWordsEnglish.extend(punctuation)


def check_validity(word):
    if(word not in stopWordsEnglish and not word.isnumeric()):
        return True
    return False


def format_input(test):
    tokenized_sentence = word_tokenize(test)
    final = list(filter(check_validity, tokenized_sentence))
    return d2v_model.infer_vector(final)

keras_model = keras.models.load_model('model_keras')


def predict_input(input):
    formatted_input = format_input(input)
    formatted_input = np.array(formatted_input)
    formatted_input = formatted_input.reshape(1, 100)
    if keras_model.predict([formatted_input])[0][0] > 0.5:
        return True
    return False

sys.modules[__name__] = predict_input
