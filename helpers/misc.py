from __future__ import print_function

import re
import string
import nltk
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords
from pandas import Series, DataFrame
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split as sk_train_test_split
from average_precision import mapk
import itertools
import copy
import igraph

def split_cell(df, col1, col2, dtype = int):
    """Helper function to break a cell consisting multiple entries into several rows."""
    df_ = pd.concat([Series(row[col1], row[col2].split(' ')) for _, row in df.iterrows()]).reset_index()
    df_.columns = [col2, col1]
    df_[col2] = df_[col2].astype(dtype)
    return df_

def train_test_split(data, info, **args):
    """Split data and info into training set and test set."""
    cv = args.get('cv', None)
    if cv is None:
        cv = 1
    random_state = args.get('random_state', None)
    if cv > 1:
        rs = ShuffleSplit(n_splits = cv, test_size = 1.0/cv, random_state = random_state)
    else:
        rs = ShuffleSplit(n_splits = 1, test_size = 0.1, random_state = random_state)
    indices = data['mids'].apply(lambda x: list(rs.split(x.split(' '))))
    train_test = []
    for i in xrange(cv):
        train = data.copy()
        test = data.copy()
        train['mids'] = train['mids'].apply(lambda x: x.split(' '))
        test['mids'] = test['mids'].apply(lambda x: x.split(' '))

        train['indices'] = indices.apply(lambda x: x[i][0])
        test['indices'] = indices.apply(lambda x: x[i][1])

        train['tmp'] = train.apply(lambda x: map(lambda y: x['mids'][y], x['indices']), axis = 1)
        test['tmp'] = test.apply(lambda x: map(lambda y: x['mids'][y], x['indices']), axis = 1)

        train.drop(['mids', 'indices'], axis = 1, inplace = True)
        test.drop(['mids', 'indices'], axis = 1, inplace = True)

        train.rename(columns={'tmp': 'mids'}, inplace = True)
        test.rename(columns={'tmp': 'mids'}, inplace = True)

        train['mids'] = train['mids'].apply(lambda x: ' '.join(x))
        test['mids'] = test['mids'].apply(lambda x: ' '.join(x))
        X_train, y_train = make_X_y(train, info)
        X_test, y_test = make_X_y(test, info)
        train_test.append((X_train, y_train, X_test, y_test))
    return train_test

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

def clean_text_simple(text, remove_stopwords=True, pos_filtering=True, stemming=True):

    original_text = text
    punct = string.punctuation.replace('-', '')
    # convert to lower case
    text = text.lower()
    # remove punctuation (preserving intra-word dashes)
    cond = '[' + re.escape(punct) + ']+'
    text = re.sub(cond, ' ', text)
    text = re.sub('(\s+-|-\s+)', ' ', text)
    # strip extra white space
    text = re.sub('-{2,}', ' ', text)
    text = re.sub('\s+', ' ', text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize (split based on whitespace)
    tokens = text.split(' ')
    tokens = filter(lambda x: len(x) > 0, tokens)
    if pos_filtering == True and len(tokens) > 0:
        # apply POS-tagging
        tagged_tokens = pos_tag(tokens)
        # retain only nouns and adjectives
        tokens_keep = []
        for i in range(len(tagged_tokens)):
            item = tagged_tokens[i]
            if (
                item[1] == 'NN' or
                item[1] == 'NNS' or
                item[1] == 'NNP' or
                item[1] == 'NNPS' or
                item[1] == 'JJ' or
                item[1] == 'JJS' or
                item[1] == 'JJR'
                ):
                    tokens_keep.append(item[0])
        tokens = tokens_keep
    if remove_stopwords:
        stpwds = set(stopwords.words('english'))
        # remove stopwords
        tokens = [token for token in tokens if token not in stpwds]
    if stemming:
        stemmer = nltk.stem.PorterStemmer()
        # apply Porter's stemmer
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    return tokens

def terms_to_graph(terms, w):
    # This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox']
    # Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'

    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in xrange(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i - w + 1):(i + 1)]

        # edges to try
        candidate_edges = []
        for p in xrange(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    g.add_vertices(sorted(set(terms)))

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertice weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding window
    g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return (g)


def unweighted_k_core(g):
    # work on clone of g to preserve g
    gg = copy.deepcopy(g)

    # initialize dictionary that will contain the core numbers
    cores_g = dict(zip(gg.vs['name'], [0] * len(gg.vs)))

    i = 0

    # while there are vertices remaining in the graph
    while len(gg.vs) > 0:
        # while there is a vertex with degree less than i
        while [deg for deg in gg.strength() if deg <= i]:
            index = [ind for ind, deg in enumerate(gg.strength()) if deg <= i][0]
            # assign i to the vertices core numbers
            cores_g[gg.vs[index]['name']] = i
            gg.delete_vertices(index)

        i += 1

    return cores_g

def clean(X, col = 'body', cleaner = clean_text_simple, join = True):
    X_cleaned = X.copy()
    X_cleaned[col] = X_cleaned[col].apply(cleaner)
    if join:
        X_cleaned[col] = X_cleaned[col].apply(lambda x: ' '.join(x))
    return X_cleaned

def compute_node_centrality(graph):
    # degree
    degrees = graph.degree()
    degrees = [round(float(degree)/(len(graph.vs)-1),5) for degree in degrees]

    # weighted degree
    w_degrees = graph.strength(weights=graph.es["weight"])
    w_degrees = [round(float(degree)/(len(graph.vs)-1),5) for degree in w_degrees]

    # closeness
    closeness = graph.closeness(normalized=True)
    closeness = [round(value,5) for value in closeness]

    # weighted closeness
    w_closeness = graph.closeness(normalized=True, weights=graph.es["weight"])
    w_closeness = [round(value,5) for value in w_closeness]

    return(zip(graph.vs["name"],degrees,w_degrees,closeness,w_closeness))
