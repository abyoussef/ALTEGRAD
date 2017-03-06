from __future__ import print_function
import os
import numpy as np
import pandas as pd
from helpers.misc import train_test_split, score, make_X_y, write_to_file
from methods.baseline import baseline
from methods.tfidf_centroid import tfidf_centroid
from methods.twidf_centroid import twidf_centroid

def test(method, cv = None):
    path_to_data = 'Data/'

    data = pd.read_csv(os.path.join(path_to_data + 'training_set.csv'), sep=',', header=0)

    info = pd.read_csv(os.path.join(path_to_data + 'training_info.csv'), sep=',', header=0)

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

    training = pd.read_csv(os.path.join(path_to_data + 'training_set.csv'), sep=',', header=0)
    training_info = pd.read_csv(os.path.join(path_to_data + 'training_info.csv'), sep=',', header=0)
    X_train, y_train = make_X_y(training, training_info)

    test = pd.read_csv(os.path.join(path_to_data + 'test_set.csv'), sep=',', header=0)
    test_info = pd.read_csv(os.path.join(path_to_data + 'test_info.csv'), sep=',', header=0)
    X_test, _ = make_X_y(test, test_info)

    y_pred = method(X_train, y_train, X_test)

    write_to_file(y_pred, os.path.join(path_to_data, method.__name__ + '.csv'))

if __name__ == '__main__':
    #test(tfidf_centroid, cv = 3)
    submission(twidf_centroid)
