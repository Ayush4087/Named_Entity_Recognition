import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
import nltk
import pickle
from utility import *

df_train = pd.read_csv('CRF_Dataset/Train_NER_CRF.csv', encoding = "ISO-8859-1") ## pass the path of train CSV file
# df_train.head()

df_test = pd.read_csv('CRF_Dataset/Test_NER_CRF.csv', encoding = "ISO-8859-1")
# df_test.head()

df_train = df_train.fillna(method='ffill')
df_train['Sentence'].nunique(), df_train.Word.nunique(), df_train.Tag.nunique()

df_test = df_test.fillna(method='ffill')
df_test['Sentence'].nunique(), df_test.Word.nunique(), df_test.Tag.nunique()


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist(), 
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

getter_train = SentenceGetter(df_train)
sentences = getter_train.sentences

getter_test = SentenceGetter(df_test)
sentences_test = getter_test.sentences


X_train = [sent2features(s) for s in sentences]
Y_train = [sent2labels(s) for s in sentences]

X_test = [sent2features(s) for s in sentences_test]
Y_test = [sent2labels(s) for s in sentences_test]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.05440064024732877,
    c2=0.002549493157422235,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, Y_train)

labels = list(crf.classes_)
labels.remove('O')

Y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(Y_test, Y_pred, labels = labels)) ## it will print the confusion matrix


filename = 'CRF_MODEL.sav'
pickle.dump(crf, open(filename, 'wb'))  ## saving the model as pickle file




