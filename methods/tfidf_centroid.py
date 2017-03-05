import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.misc import split_cell
from methods.baseline import freq

def rerank(X_train, y_train, X_test, y_pred):
    counts = freq(X_train, y_train)

    snd_mid_rec = pd.merge(left = X_test[['sender', 'mid']], right = y_pred, on = 'mid', how = 'inner')
    snd_rec = split_cell(snd_mid_rec[['sender', 'recipients']], 'sender', 'recipients', str)
    y_pred = pd.merge(left = snd_rec, right = counts, on = ['sender', 'recipients'], how = 'left')

    y_pred.set_index(['sender', 'recipients'], inplace = True)
    y_pred = y_pred['counts']
    g = y_pred.groupby(level=0, group_keys=False)
    tmp = g.nlargest(10)
    tmp = tmp.reset_index().drop('counts', axis = 1)
    tmp = tmp.groupby('sender')['recipients'].apply(lambda x: ' '.join(x)).reset_index()
    y_pred = pd.merge(left = X_test, right = tmp, on = 'sender', how = 'inner')[['mid', 'recipients']]
    return y_pred

def tfidf_centroid(X_train, y_train, X_test):
    # Data frame containing mid and recipients
    mid_rec = split_cell(y_train, 'mid', 'recipients', str)
    # Data frame containing sender, mid and recipients
    snd_mid_rec = pd.merge(left = X_train[['sender', 'mid']], right = mid_rec, on = 'mid', how = 'inner')
    # Data frame for final prediction
    y_pred = DataFrame()
    for sender, emails in X_train.groupby('sender'):
        # Loop over sender
        # For current sender, compute the TF-IDF matrix
        clf = TfidfVectorizer(stop_words='english')
        tfidf = clf.fit_transform(emails['body'])
        ## L-1 normalization
        tfidf /= (tfidf.sum(axis = 1) + 1e-15)
        tfidf = csr_matrix(tfidf)

        # Map the recipient to a TF-IDF vector
        df = DataFrame()
        df['mid'] = emails['mid'].reset_index(drop=True)
        df['tfidf'] = Series([tfidf[i] for i in xrange(tfidf.shape[0])])

        # Sum the TF-IDF vector for a given recipients
        # As a result, for each sender-recipient pair, we have a TF-IDF vector
        df = pd.merge(left = df, right = snd_mid_rec, on = 'mid', how = 'inner')
        df.drop('mid', axis = 1, inplace = True)
        df = df.groupby(['sender', 'recipients'])['tfidf'].sum()
        df = df.reset_index()

        # List of recipients
        recs = df['recipients'].values

        # Final TF-IDF matrix
        tfidf = vstack(df['tfidf'].values)

        # Emails in test set
        emails_test = X_test[X_test['sender'] == sender]

        # TF-IDF matrix of test email using the same transformation as training
        tfidf_test = clf.transform(emails_test['body'])

        # Calculate the simularity
        sim = cosine_similarity(tfidf_test, tfidf)

        # Get the top 10 recipients by cosine simulatiry
        top10 = np.fliplr(np.argsort(sim, axis = 1))[:, :10]
        top10 = recs[top10]
        top10 = map(lambda x: ' '.join(x), top10)

        # Prediction for current sender
        df = DataFrame()
        df['mid'] = emails_test['mid']
        df['recipients'] = Series(top10, index = emails_test.index)

        # Add to final prediction
        y_pred = pd.concat([y_pred, df])
    #y_pred = rerank(X_train, y_train, X_test, y_pred)
    return y_pred
