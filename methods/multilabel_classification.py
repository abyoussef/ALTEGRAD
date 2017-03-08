import pandas as pd
from pandas import Series, DataFrame
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from helpers.misc import split_cell
from methods.baseline import baseline_score

def binarize_recipients(recipients):
    recipients = split_cell(recipients, 'mid', 'recipients', str)
    recipients['is_recipient'] = 1
    recipients = recipients.pivot('mid', 'recipients', 'is_recipient').fillna(0)
    return recipients
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

def multilabel_classification(X_train, y_train, X_test):
    scores_freq = baseline_score(X_train, y_train, X_test)
    scores_freq_sum = scores_freq.groupby('mid')['score'].sum().reset_index()
    scores_freq = pd.merge(left = scores_freq, right = scores_freq_sum, on = 'mid')
    scores_freq['score'] = scores_freq['score_x'] / scores_freq['score_y']
    scores_freq = scores_freq[['mid', 'recipients', 'score']]

    scores_mc = multilabel_classification_score(X_train, y_train, X_test)
    scores_mc_sum = scores_mc.groupby('mid')['score'].sum().reset_index()
    scores_mc = pd.merge(left = scores_mc, right = scores_mc_sum, on = 'mid')
    scores_mc['score'] = scores_mc['score_x'] / scores_mc['score_y']
    scores_mc = scores_mc[['mid', 'recipients', 'score']]

    scores_freq.rename(columns = {'score': 'score_freq'}, inplace = True)
    scores_mc.rename(columns = {'score': 'score_mc'}, inplace = True)
    scores = pd.merge(left = scores_freq, right = scores_mc, on = ['mid', 'recipients'], how = 'inner')

    scores['score'] = scores['score_freq'] + scores['score_mc']

    scores.sort_values(['mid', 'score'], inplace = True, ascending = [True, False])

    y_pred = scores[['mid', 'recipients']]
    y_pred = y_pred.groupby('mid').head(10).reset_index(drop = True)
    y_pred = y_pred.groupby(['mid'])['recipients'].apply(lambda x: ' '.join(x)).reset_index()
    return y_pred
