import pandas as pd
from pandas import Series, DataFrame
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
from helpers.misc import split_cell, top_k_score
from helpers.gow import TwidfVectorizer

def remove_empty_graphs(X, y = None, col ='body', w = 4):
    if y is not None:
        X = pd.merge(left = X, right = y, on = 'mid', how = 'inner')
    else:
        X = X.copy()
    X[col] = X[col].apply(lambda x: x.split(' '))
    X = X[X[col].apply(lambda x: len(set(x))) >= 2].reset_index(drop = True)
    X = X[X[col].apply(len) >= w].reset_index(drop = True)
    X[col] = X[col].apply(lambda x: ' '.join(x))
    if y is not None:
        return X.drop('recipients', axis = 1), X[['mid', 'recipients']]
    return X

def twidf_score(X_train, y_train, X_test):
    # Data frame containing mid and recipients
    mid_rec = split_cell(y_train, 'mid', 'recipients', str)
    # Data frame containing sender, mid and recipients
    snd_mid_rec = pd.merge(left = X_train[['sender', 'mid']], right = mid_rec, on = 'mid', how = 'inner')
    # Data frame for final prediction
    scores = DataFrame()

    clf = TwidfVectorizer()

    df = DataFrame(index=X_train.index)
    df['mid'] = X_train['mid']
    tfidf = clf.fit_transform(X_train['body'])
    df['tfidf'] = Series([tfidf[i] for i in xrange(tfidf.shape[0])])
    df = pd.merge(left=df, right=snd_mid_rec, on='mid', how='inner')
    df = df.groupby(['sender', 'recipients'])['tfidf'].sum().reset_index()

    df_test = DataFrame(index=X_test.index)
    df_test['mid'] = X_test['mid']
    tfidf_test = clf.transform(X_test['body'])
    df_test['tfidf'] = Series([tfidf_test[i] for i in xrange(tfidf_test.shape[0])])
    df_test = pd.merge(left = df_test, right = X_test[['sender', 'mid']], how = 'inner', on = 'mid')

    for sender, _ in X_test.groupby('sender'):
        # Loop over sender
        # For current sender, compute the TF-IDF matrix
        df_this = df[df['sender'] == sender].copy()
        df_test_this = df_test[df_test['sender'] == sender].copy()

        recs = df_this['recipients'].values

        tfidf = vstack(df_this['tfidf'].values)
        tfidf_test = vstack(df_test_this['tfidf'].values)

        sim = cosine_similarity(tfidf_test, tfidf)

        tmp = DataFrame(sim, columns = recs, index = df_test_this['mid'])
        tmp = DataFrame(tmp.stack()).reset_index()
        tmp.columns = ['mid', 'recipients', 'score']
        scores = pd.concat([scores, tmp]).reset_index(drop = True)
    return scores

def twidf(X_train, y_train, X_test):
    X_train_cleaned, y_train_cleaned = remove_empty_graphs(X_train, y_train)
    X_test_cleaned = remove_empty_graphs(X_test)

    scores = twidf_score(X_train_cleaned, y_train_cleaned, X_test_cleaned)
    y_pred = top_k_score(scores)
    y_pred = pd.merge(left = X_test, right = y_pred, on = 'mid', how = 'left')[['mid', 'recipients']]
    y_pred.fillna('', inplace = True)
    return y_pred
