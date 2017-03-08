import pandas as pd
from pandas import DataFrame, Series
from scipy.sparse import vstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from helpers.gow import TwidfVectorizer
from helpers.misc import split_cell, binarize_recipients


def multilabel_classification_score(X_train, y_train, X_test):
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(RandomForestClassifier()))])
    scores = DataFrame()
    for sender, emails in X_train.groupby('sender'):
        recipients = pd.merge(left = emails, right = y_train, on = 'mid', how = 'inner')[['mid', 'recipients']]
        recipients = binarize_recipients(recipients)
        recipients.sort_index(inplace=True)
        emails = emails.sort_values('mid')
        emails['body'].fillna('', inplace = True)
        classifier.fit(emails['body'], recipients)

        emails_test = X_test[X_test['sender'] == sender].copy()
        emails_test['body'].fillna('', inplace = True)
        if len(recipients.columns) == 1:
            scores_this = DataFrame(classifier.predict_proba(emails_test['body'])[:, 0], columns = recipients.columns, index = emails_test['mid'])
        else:
            scores_this = DataFrame(classifier.predict_proba(emails_test['body']), columns = recipients.columns, index = emails_test['mid'])
        scores_this = scores_this.stack()
        scores_this = scores_this.reset_index()
        scores_this.rename(columns={0: 'score'}, inplace = True)
        scores = pd.concat([scores, scores_this]).reset_index(drop = True)
    return scores


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


def tfidf_centroid_score(X_train, y_train, X_test):
    # Data frame containing mid and recipients
    mid_rec = split_cell(y_train, 'mid', 'recipients', str)
    # Data frame containing sender, mid and recipients
    snd_mid_rec = pd.merge(left = X_train[['sender', 'mid']], right = mid_rec, on = 'mid', how = 'inner')
    # Data frame for final prediction
    scores = DataFrame()
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

        df = DataFrame(sim, columns = recs, index = emails_test.mid)
        df = df.stack().reset_index()
        df.columns = ['mid', 'recipients', 'score']
        scores = pd.concat([scores, df]).reset_index(drop = True)
    return scores

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

def baseline_score(X_train, y_train, X_test):
    X_train = X_train[['sender', 'mid']]
    y_train = split_cell(y_train, 'mid', 'recipients', dtype = str)
    train = pd.merge(left = X_train, right = y_train, on = 'mid', how = 'inner')
    counts = train.groupby(['sender', 'recipients'])['mid'].count()
    counts = counts.reset_index(name = 'score')
    scores = pd.merge(left = X_test, right = counts, on = 'sender')[['mid', 'recipients', 'score']]
    return scores