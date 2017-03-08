import pandas as pd
import numpy as np
import igraph
from helpers.misc import split_cell
from sklearn.feature_extraction.text import CountVectorizer

class CommunicationGraph():
    def __init__(self):
        self.g = igraph.Graph(directed=True)
        self.count_vectorizer = CountVectorizer()

    def build_graph(self, X, y):

        X = X.copy()
        y = y.copy()

        # Compute the word counts
        word_counts = self.count_vectorizer.fit_transform(X['body'])
        X['wcount'] = pd.Series([word_counts[i] for i in xrange(word_counts.shape[0])])

        # Merge emails with their recipients
        mid_rec = split_cell(y, 'mid', 'recipients', str)
        snd_mid_wcount_rec = pd.merge(left=X[['sender', 'mid', 'wcount']], right=mid_rec, on='mid', how='inner')

        # Extract edges from sender to recipients
        # Each edge will store the number of mails exchanged (freq) and the sum of word counts (wcount)
        from_to_freq = {}
        from_to_wcount = {}
        for index, row in snd_mid_wcount_rec.iterrows():
            if index % 10000 == 0:
                print index
            try_edge = (row['sender'], row['recipients'])
            if try_edge[1] != try_edge[0]:
                if try_edge in from_to_freq:
                    from_to_freq[try_edge] += 1
                    from_to_wcount[try_edge] += row['wcount']
                else:
                    from_to_freq[try_edge] = 1
                    from_to_wcount[try_edge] = row['wcount']

        # Extract vertices
        all_users = np.unique(np.concatenate((X['sender'].values, mid_rec['recipients'].unique())))

        # Fill the graph
        self.g.add_vertices(sorted(all_users))
        self.g.add_edges(from_to_freq.keys())

        self.g.es['freq'] = from_to_freq.values()
        self.g.es['wcount'] = from_to_wcount.values()

        self.g.vs['indegree'] = self.g.strength(weights=from_to_freq.values(), mode='in')
        self.g.vs['outdegree'] = self.g.strength(weights=from_to_freq.values(), mode='out')








