import pandas as pd
from pandas import Series, DataFrame

from helpers.misc import split_cell, top_k_score

def freq(X_train, y_train):
    X_train = X_train[['sender', 'mid']]
    y_train = split_cell(y_train, 'mid', 'recipients', dtype = str)
    train = pd.merge(left = X_train, right = y_train, on = 'mid', how = 'inner')
    counts = train.groupby(['sender', 'recipients'])['mid'].count()
    return counts.reset_index(name = 'counts')

def baseline_score(X_train, y_train, X_test):
    counts = freq(X_train, y_train)
    counts.rename(columns = {'counts':'score'}, inplace = True)
    scores = pd.merge(left = X_test, right = counts, on = 'sender')[['mid', 'recipients', 'score']]
    return scores

def baseline(X_train, y_train, X_test):
    scores = baseline_score(X_train, y_train, X_test)
    y_pred = top_k_score(scores)
    return y_pred