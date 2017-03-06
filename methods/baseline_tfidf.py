import pandas as pd
from pandas import Series, DataFrame
from methods.baseline import baseline_score
from methods.tfidf import tfidf_score

def baseline_tfidf(X_train, y_train, X_test):
    """Use baseline method but sort with respect to TF-IDF similarity."""
    scores_freq = baseline_score(X_train.copy(), y_train.copy(), X_test.copy())
    scores_sim = tfidf_score(X_train.copy(), y_train.copy(), X_test.copy())
    scores_freq.rename(columns = {'score': 'score_freq'}, inplace = True)
    scores_sim.rename(columns = {'score': 'score_sim'}, inplace = True)
    scores = pd.merge(left = scores_freq, right = scores_sim, on = ['mid', 'recipients'], how = 'inner')

    scores.sort_values(['mid', 'score_freq'], inplace = True, ascending = [True, False])
    scores = scores.groupby('mid').head(10).reset_index(drop = True)

    y_pred = scores.sort_values(['mid', 'score_sim'], ascending = [True, False])[['mid', 'recipients']]
    y_pred = y_pred.groupby('mid')['recipients'].apply(lambda x: ' '.join(x)).reset_index()
    return y_pred