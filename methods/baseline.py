import pandas as pd
from pandas import Series, DataFrame

from helpers.misc import split_cell

#TODO: extract a ``score'' function
def freq(X_train, y_train):
    X_train = X_train[['sender', 'mid']]
    y_train = split_cell(y_train, 'mid', 'recipients', dtype = str)
    train = pd.merge(left = X_train, right = y_train, on = 'mid', how = 'inner')
    counts = train.groupby(['sender', 'recipients'])['mid'].count()
    return counts.reset_index(name = 'counts')

def baseline(X_train, y_train, X_test):
    counts = freq(X_train, y_train)
    counts.set_index(['sender', 'recipients'], inplace = True)
    counts = counts['counts']
    g = counts.groupby(level=0, group_keys=False)
    tmp = g.nlargest(10)
    tmp = tmp.reset_index().drop('counts', axis = 1)
    tmp = tmp.groupby('sender')['recipients'].apply(lambda x: ' '.join(x)).reset_index()
    y_pred = pd.merge(left = X_test, right = tmp, on = 'sender', how = 'inner')[['mid', 'recipients']]
    return y_pred