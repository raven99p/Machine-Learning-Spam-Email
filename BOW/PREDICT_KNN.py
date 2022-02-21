import pandas as pd
import pickle
import sys

log_reg_model = pickle.load(
    open("./saved_models/model_knn.sav", 'rb'))
vectorizer = pickle.load(open('./saved_models/vectorizer.pkl', 'rb'))


def predict_input(input):
    formatted_input = vectorizer.transform([input]).toarray()
    if log_reg_model.predict([formatted_input[0]])[0]:
        print('true')
        return True
    print('false')
    return False

sys.modules[__name__] = predict_input
