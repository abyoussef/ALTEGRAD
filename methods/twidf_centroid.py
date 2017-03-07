import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from helpers.misc import split_cell
from helpers.clean import clean
from helpers.gow import TwidfVectorizer
from methods.baseline import freq

def remove_empty_graphs(df, col = 'body', w = 4):
    df = df.copy()
    df[col] = df[col].apply(lambda x: x.split(' '))
    df = df[df[col].apply(lambda x: len(set(x))) >= 2].reset_index(drop = True)
    df = df[df[col].apply(len) >= w].reset_index(drop = True)
    df[col] = df[col].apply(lambda x: ' '.join(x))
    return df

def twidf_centroid(X_train, y_train, X_test):
    # Data frame containing mid and recipients
    mid_rec = split_cell(y_train, 'mid', 'recipients', str)
    # Data frame containing sender, mid and recipients
    snd_mid_rec = pd.merge(left = X_train[['sender', 'mid']], right = mid_rec, on = 'mid', how = 'inner')
    # Data frame for final prediction
    y_pred = DataFrame()

    X_train_cleaned = clean(X_train)
    X_test_cleaned = clean(X_test)

    X_train_cleaned = remove_empty_graphs(X_train_cleaned)
    X_test_cleaned = remove_empty_graphs(X_test_cleaned)

    for sender, emails in X_train_cleaned.groupby('sender'):
        # Loop over sender
        # For current sender, compute the TF-IDF matrix
        clf = TwidfVectorizer()
        twidf = clf.fit_transform(emails['body'])
        ## L-1 normalization
        twidf /= (twidf.sum(axis = 1) + 1e-15)
        twidf = csr_matrix(twidf)

        # Map the recipient to a TF-IDF vector
        df = DataFrame()
        df['mid'] = emails['mid'].reset_index(drop=True)
        df['tfidf'] = Series([twidf[i] for i in xrange(twidf.shape[0])])

        # Sum the TF-IDF vector for a given recipients
        # As a result, for each sender-recipient pair, we have a TF-IDF vector
        df = pd.merge(left = df, right = snd_mid_rec, on = 'mid', how = 'inner')
        df.drop('mid', axis = 1, inplace = True)
        df = df.groupby(['sender', 'recipients'])['tfidf'].sum()
        df = df.reset_index()

        # List of recipients
        recs = df['recipients'].values

        # Final TF-IDF matrix
        twidf = vstack(df['tfidf'].values)

        # Emails in test set
        emails_test = X_test_cleaned[X_test_cleaned['sender'] == sender]

        # TF-IDF matrix of test email using the same transformation as training
        twidf_test = clf.transform(emails_test['body'])

        # Calculate the simularity
        sim = cosine_similarity(twidf_test, twidf)

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
    y_pred = pd.merge(left = X_test, right = y_pred, on = 'mid', how = 'left')[['mid', 'recipients']]
    y_pred['recipients'].fillna('', inplace = True)
    return y_pred
