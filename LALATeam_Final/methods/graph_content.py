import pandas as pd
import time
import numpy as np
from scipy.sparse import vstack
from helpers.com_graph import CommunicationGraph


def graph_content(X_train, y_train, X_test, lamb=0.6, gamma=0.2, beta=0.2, mu=0.3):

    # Fix empty documents
    X_train.fillna('', inplace=True)
    X_test.fillna('', inplace=True)

    print 'building graph...'
    graph = CommunicationGraph()
    graph.build_graph(X_train, y_train)

    # compute once and for all predictions these quantities used later in the computation of email likelihood
    edges = graph.g.es
    p_w = vstack([e['wcount'] for e in edges], dtype=float).sum(axis=0)
    p_w = p_w / p_w.sum()
    sum_freq = np.sum([e['freq'] for e in edges])

    # to avoid numerical errors
    eps = 1e-20

    y_pred = pd.DataFrame(columns=['mid', 'recipients'])

    # FOR DEBUG
    n_senders = len(np.unique(X_test.sender))
    k = 0
    for sender, df in X_test.groupby('sender'):

        print 'predicting for sender ',sender, '(',k, '/', n_senders,')'
        k += 1

        sender_id = graph.g.vs.find(sender).index

        # sequence of edges from and to sender
        edges_s = edges.select(_from=sender_id)
        edges_to_s = edges.select(_to=sender_id)

        # list of vertex id corresponding to potential recipients
        recs_id = [e.target for e in edges_s]
        recs_names = [graph.g.vs[e.target]['name'] for e in edges_s]

        # emails to be predicted (converted to word counts)
        emails = graph.count_vectorizer.transform(df['body'].values)
        mids = df['mid'].values

        # these quantities will only be computed once for this sender as their dont depend on the content of the emails.
        log_p_s_knowing_r = np.zeros(len(recs_id))
        log_p_r = np.zeros(len(recs_id))

        first_email = True
        for mid, email in zip(mids, emails):

            # special case to prevent bug in empty graph
            if len(recs_names) == 0:
                y_pred.loc[int(mid)] = [int(mid), sender]
                continue

            log_p_e_knowing_r_s = np.zeros(len(recs_id))
            for i, rec in enumerate(recs_id):
                edges_to_r = edges.select(_to=rec)
                edges_from_r = edges.select(_from=rec)
                edges_from_s_to_r = edges_to_r.select(_from=sender_id)
                edges_from_r_to_s = edges_from_r.select(_to=sender_id)

                # EMAIL LIKELIHOOD
                # compute for all words at the same time the conditional probabilities
                p_w_knowing_r = vstack([e['wcount'] for l in [edges_to_r, edges_from_r] for e in l], dtype=float).sum(
                    axis=0)
                p_w_knowing_r = p_w_knowing_r / (p_w_knowing_r.sum() + eps)
                p_w_knowing_r_s = vstack([e['wcount'] for l in [edges_from_s_to_r, edges_from_r_to_s] for e in l],
                                         dtype=float).sum(axis=0)
                p_w_knowing_r_s = p_w_knowing_r_s / (p_w_knowing_r_s.sum() + eps)

                # compute the email conditional probability
                words = email.indices
                counts = email.data
                p_e_knowing_r_s = lamb * p_w_knowing_r_s[0, words]
                p_e_knowing_r_s += gamma * p_w_knowing_r[0, words]
                p_e_knowing_r_s += beta * p_w[0, email.indices]
                p_e_knowing_r_s = np.power(p_e_knowing_r_s, counts)
                log_p_e_knowing_r_s[i] = np.sum(np.log(p_e_knowing_r_s))

                if (first_email):
                    # RECIPIENT LIKELIHOOD
                    p_r = np.sum([e['freq'] for e in edges_to_r], dtype=float)
                    p_r /= sum_freq
                    log_p_r[i] = np.log(p_r)

                    # SENDER LIKELIHOOD
                    p_s_knowing_r = np.sum([e['freq'] for e in edges_from_s_to_r], dtype=float)
                    p_s_knowing_r /= np.sum([e['freq'] for e in edges_to_r])
                    log_p_s_knowing_r[i] = np.log(p_s_knowing_r)

            # Compute scores
            scores = (1 - mu) * (log_p_r + log_p_s_knowing_r) + mu * log_p_e_knowing_r_s
            scores_sorted, names_sorted = (list(t) for t in zip(*sorted(zip(scores, recs_names), reverse=True)))

            # Store prediction for current email
            y_pred.loc[int(mid)] = [int(mid), ' '.join(names_sorted[:10])]
            first_email = False

    y_pred['mid'] = y_pred['mid'].astype(int)

    return y_pred







