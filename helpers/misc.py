from __future__ import print_function

import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split as sk_train_test_split
from average_precision import mapk

def split_cell(df, col1, col2, dtype = int):
    """Helper function to break a cell consisting multiple entries into several rows."""
    df_ = pd.concat([Series(row[col1], row[col2].split(' ')) for _, row in df.iterrows()]).reset_index()
    df_.columns = [col2, col1]
    df_[col2] = df_[col2].astype(dtype)
    return df_

def train_test_split(data, info, **args):
    """Split data and info into training set and test set."""
    train = data.drop('mids', axis = 1)
    test = data.drop('mids', axis = 1)

    tmp = data['mids'].apply(lambda x: sk_train_test_split(x.split(' '), **args))

    train['mids'] = tmp.apply(lambda x: ' '.join(x[0]))
    test['mids'] = tmp.apply(lambda x: ' '.join(x[1]))

    X_train, y_train = make_X_y(train, info)
    X_test, y_test = make_X_y(test, info)

    return X_train, y_train, X_test, y_test

def make_X_y(data, info):
    data = split_cell(data, 'sender', 'mids')
    data = pd.merge(left=data, right=info, left_on='mids', right_on='mid', how='left')
    data.drop(['mids'], axis=1, inplace=True)
    try:
        X = data.drop('recipients', axis=1)
        y = data[['mid', 'recipients']].copy()
        y['recipients'] = y['recipients'].apply(lambda x: ' '.join(filter(lambda y: '@' in y, x.split(' '))))
    except ValueError:
        X = data.copy()
        y = None
    return X, y

def score(y_true, y_pred, k=10):
    """Return MAP@10 score for true and predicted recipients. """
    y_true_sorted = y_true.sort_values('mid')
    y_pred_sorted = y_pred.sort_values('mid')
    y_true_sorted = y_true_sorted['recipients'].apply(lambda x: x.split(' ')).tolist()
    y_pred_sorted = y_pred_sorted['recipients'].apply(lambda x: x.split(' ')).tolist()
    return mapk(y_true_sorted, y_pred_sorted, k)

def write_to_file(y_pred, filename):
    """Write predicted recipients to file."""
    with open(filename, 'w') as f:
        f.write('mid,recipients\n')
        for _, row in y_pred.iterrows():
            f.write(str(row['mid']) + ',' + row['recipients'] + '\n')

if __name__ == '__main__':
    import os
    path_to_data = 'Data/'
    data = pd.read_csv(os.path.join(path_to_data + 'training_set.csv'), sep=',', header=0)
    info = pd.read_csv(os.path.join(path_to_data + 'training_info.csv'), sep=',', header=0)
    X_train, y_train, X_test, y_test = train_test_split(data, info, test_size = 0.1, random_state = 0)
    print(score(y_test, y_test))
