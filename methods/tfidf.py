import pandas as pd
from pandas import Series, DataFrame
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from helpers.misc import split_cell, top_k_score

def tfidf_score(X_train, y_train, X_test):
    # Data frame containing mid and recipients
    mid_rec = split_cell(y_train, 'mid', 'recipients', str)
    # Data frame containing sender, mid and recipients
    snd_mid_rec = pd.merge(left = X_train[['sender', 'mid']], right = mid_rec, on = 'mid', how = 'inner')
    # Data frame for final prediction
    scores = DataFrame()

    clf = TfidfVectorizer()

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

def tfidf(X_train, y_train, X_test):
    scores = tfidf_score(X_train, y_train, X_test)
    y_pred = top_k_score(scores)
    return y_pred
