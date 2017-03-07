import itertools
import pandas as pd
from pandas import Series, DataFrame
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix, vstack
from helpers.misc import clean, terms_to_graph, compute_node_centrality

class TwidfVectorizer():
    def __init__(self, centrality = 'degree', b = 0.03):
        self.centrality = centrality
        self.b = b

    def df2graph(self, X, w = 4):
        graphs = X.apply(lambda x: terms_to_graph(x, w))
        return graphs

    def process_doc(self, g):
        metrics = compute_node_centrality(g)
        df_this = DataFrame(metrics, columns = ['name', 'degree', 'w_degree', 'closeness', 'w_closeness'])
        df_this.set_index('name', inplace=True)
        df = pd.merge(left = self.df_ref, right = df_this[[self.centrality]], how = 'left', left_index=True, right_index=True)
        df.fillna(0, inplace = True)
        return csr_matrix(df[self.centrality])

    def fit(self, X):
        X = X.apply(lambda x: x.split(' '))
        self.vocabulary = list(set(list(itertools.chain.from_iterable(X.values))))
        self.df_ref = DataFrame(self.vocabulary, columns = ['name'])
        self.df_ref.set_index('name', inplace = True)
        self._is_fitted = True

    def transform(self, X):
        X = X.apply(lambda x: x.split(' '))

        graphs = self.df2graph(X)

        tw = []
        for _, g in graphs.iteritems():
            tw.append(self.process_doc(g))
        tw = vstack(tw)

        self.graphs = graphs
        n_terms_per_doc = tw.sum(axis = 1)
        avg_terms_per_doc = n_terms_per_doc.mean()
        tw = csr_matrix(tw / (1 - self.b + self.b*n_terms_per_doc/avg_terms_per_doc))

        clf = TfidfTransformer()
        return clf.fit_transform(tw)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
