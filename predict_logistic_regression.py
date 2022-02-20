import pandas as pd
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import sys
from nltk.stem import PorterStemmer
ps = PorterStemmer()

d2v_model = Doc2Vec.load('d2vModel_log.d2v')
stopWordsEnglish = stopwords.words('english')

punctuation = [".", ";", ":",  "?", "(", ")", "[", "]", "\"",
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

model_filename = "model_logistic_regression.sav"
log_reg_model = pickle.load(open(model_filename, 'rb'))

def predict_input(input):
    formatted_input = format_input(input)
    if log_reg_model.predict([formatted_input])[0]:
        return True
    return False


sys.modules[__name__] = predict_input
