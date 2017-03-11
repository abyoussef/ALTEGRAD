import pandas as pd
from helpers.misc import top_k_score, remove_empty_graphs
from methods.score import baseline_score, multilabel_classification_score, tfidf_score, twidf_score
from methods.score import tfidf_centroid_score, word2vec_score

def baseline(X_train, y_train, X_test):
    scores = baseline_score(X_train, y_train, X_test)
    y_pred = top_k_score(scores)
    return y_pred

def tfidf(X_train, y_train, X_test):
    scores = tfidf_score(X_train, y_train, X_test)
    y_pred = top_k_score(scores)
    return y_pred

def tfidf_centroid(X_train, y_train, X_test):
    scores = tfidf_centroid_score(X_train, y_train, X_test)
    scores.set_index(['mid', 'recipients'], inplace = True)
    scores = scores['score']
    g = scores.groupby(level=0, group_keys=False)
    y_pred = g.nlargest(10)
    y_pred = y_pred.reset_index().drop('score', axis = 1)
    y_pred = y_pred.groupby('mid')['recipients'].apply(lambda x: ' '.join(x)).reset_index()
    return y_pred

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

def twidf(X_train, y_train, X_test):
    X_train_cleaned, y_train_cleaned = remove_empty_graphs(X_train, y_train)
    X_test_cleaned = remove_empty_graphs(X_test)

    scores = twidf_score(X_train_cleaned, y_train_cleaned, X_test_cleaned)
    y_pred = top_k_score(scores)
    y_pred = pd.merge(left = X_test, right = y_pred, on = 'mid', how = 'left')[['mid', 'recipients']]
    y_pred.fillna('', inplace = True)
    return y_pred

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

def word2vec(X_train, y_train, X_test):
    scores = word2vec_score(X_train, y_train, X_test)
    y_pred = top_k_score(scores)
    return y_pred

if __name__ == '__main__':
    pass
