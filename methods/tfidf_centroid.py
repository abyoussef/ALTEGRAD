import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers.misc import split_cell

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
        df['tfidf'] /= (df['tfidf'].apply(lambda x: x.sum()) + 1e-15)

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
        top10 = np.argsort(sim, axis = 1)[::-1][:, :10]
        top10 = recs[top10]
        top10 = map(lambda x: ' '.join(x), top10)

        # Prediction for current sender
        df = DataFrame()
        df['mid'] = emails_test['mid']
        df['recipients'] = Series(top10, index = emails_test.index)

        # Add to final prediction
        y_pred = pd.concat([y_pred, df])
    return y_pred
