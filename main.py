from __future__ import print_function
import os
import numpy as np
import pandas as pd
from helpers.misc import train_test_split, score, make_X_y, write_to_file
from helpers.clean import clean
from methods.method import multilabel_classification, baseline_tfidf, baseline, tfidf_centroid, twidf, tfidf

#TODO: a generic method which takes method's name(s) as input and manipulate scores

def test(method, cv = None):
    path_to_data = 'Data/'

    print('[INFO] Data loading...')
    data = pd.read_csv(os.path.join(path_to_data + 'training_set.csv'), sep=',', header=0)

    info = pd.read_csv(os.path.join(path_to_data + 'training_info.csv'), sep=',', header=0)

    print('[INFO] Performing data cleaning (tokenization, stopwords, stemming, etc.)...')
    if os.path.isfile(os.path.join(path_to_data, 'training_info_clean.csv')):
        info = pd.read_csv(os.path.join(path_to_data + 'training_info_clean.csv'), sep=',', header=0)
    else:
        info = clean(info)
        info['body'].fillna('', inplace = True)
        info.to_csv(os.path.join(path_to_data + 'training_info_clean.csv'), sep=',', index = False)

    print('[INFO] Splitting data...')
    train_test = train_test_split(data, info, test_size = 0.1, random_state = None, cv = cv)

    scores = np.empty(cv)

    if cv is None:
        cv = 1

    for i in xrange(cv):
        print('[INFO] Cross-validating fold %d/%d'%(i+1, cv))
        X_train, y_train, X_test, y_test = train_test[i]
        y_pred = method(X_train, y_train, X_test)
        scores[i] = score(y_test, y_pred)
        print('[INFO] Test score: %f' % scores[i])

    print('[INFO] Final test score:', np.mean(scores))

def submission(method):
    path_to_data = 'Data/'

    print('[INFO] Data loading...')
    training = pd.read_csv(os.path.join(path_to_data + 'training_set.csv'), sep=',', header=0)
    training_info = pd.read_csv(os.path.join(path_to_data + 'training_info.csv'), sep=',', header=0)

    print('[INFO] Performing data cleaning (tokenization, stopwords, stemming, etc.)...')
    if os.path.isfile(os.path.join(path_to_data, 'training_info_clean.csv')):
        training_info = pd.read_csv(os.path.join(path_to_data + 'training_info_clean.csv'), sep=',', header=0)
    else:
        training_info = clean(training_info)
        training_info['body'].fillna('', inplace = True)
        training_info.to_csv(os.path.join(path_to_data + 'training_info_clean.csv'), sep=',', index = False)

    print('[INFO] Data loading...')
    test = pd.read_csv(os.path.join(path_to_data + 'test_set.csv'), sep=',', header=0)
    test_info = pd.read_csv(os.path.join(path_to_data + 'test_info.csv'), sep=',', header=0)

    print('[INFO] Performing data cleaning (tokenization, stopwords, stemming, etc.)...')
    if os.path.isfile(os.path.join(path_to_data, 'test_info_clean.csv')):
        test_info = pd.read_csv(os.path.join(path_to_data + 'test_info_clean.csv'), sep=',', header=0)
    else:
        test_info = clean(test_info)
        test_info['body'].fillna('', inplace = True)
        test_info.to_csv(os.path.join(path_to_data + 'test_info_clean.csv'), sep=',', index = False)

    print('[INFO] Making training and test set...')
    X_train, y_train = make_X_y(training, training_info)
    X_test, _ = make_X_y(test, test_info)

    print('[INFO] Method %s' % (method.__name__))
    y_pred = method(X_train, y_train, X_test)

    print('[INFO] Writing to output file...')
    write_to_file(y_pred, os.path.join(path_to_data, method.__name__ + '.csv'))

    print('[INFO] Done!')

if __name__ == '__main__':
    test(multilabel_classification, cv = 5)
    #submission(multilabel_classification)
