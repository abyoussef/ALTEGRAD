import pandas as pd
import numpy as np
from scipy.sparse import vstack
from helpers.com_graph import CommunicationGraph


def graph_content(X_train, y_train, X_test, lamb=0.6, gamma=0.2, beta=0.2):
    # Fix empty documents
    X_train.fillna('', inplace=True)
    X_test.fillna('', inplace=True)

    graph = CommunicationGraph()

    print 'building graph...'
    graph.build_graph(X_train,y_train)

    y_pred = pd.DataFrame(columns=['mid', 'recipients'])
    for sender, df in X_test.groupby('sender'):
        print sender

        sender_id = graph.g.vs.find(sender).index

        # sequence of edges from sender
        edges = graph.g.es
        edges_s = edges.select(_from=sender_id)

        # list of vertex id corresponding to potential recipients
        recs_id = [e.target for e in edges_s]
        recs_names = [graph.g.vs[e.target]['name'] for e in edges_s]

        # emails to be predicted (converted to word counts)
        emails = graph.count_vectorizer.transform(df['body'].values)
        mids = df['mid'].values

        log_p_s_knowing_r = np.zeros(len(recs_id))
        log_p_r = np.zeros(len(recs_id))

        first_email = True
        for mid, email in zip(mids, emails):

            print mid
            log_p_e_knowing_r_s = np.zeros(len(recs_id))
            for i, rec in enumerate(recs_id):
                print rec

                edges_r = edges.select(_to=rec)
                edges_s_r = edges_r.select(_from=sender_id)

                # EMAIL LIKELIHOOD
                # compute for all words at the same time the conditional probabilities
                p_w = vstack([e['wcount'] for e in edges], dtype=float).sum(axis=0)
                p_w = p_w / p_w.sum()
                p_w_knowing_r = vstack([e['wcount'] for e in edges_r], dtype=float).sum(axis=0)
                p_w_knowing_r = p_w_knowing_r / p_w_knowing_r.sum()
                p_w_knowing_r_s = vstack([e['wcount'] for e in edges_s_r], dtype=float).sum(axis=0)
                p_w_knowing_r_s = p_w_knowing_r_s / p_w_knowing_r_s.sum()

                # compute the email conditional probability
                words = email.indices
                counts = email.data
                p_e_knowing_r_s = lamb * p_w_knowing_r_s[0, email.indices]
                p_e_knowing_r_s += gamma * p_w_knowing_r[0, email.indices]
                p_e_knowing_r_s += beta * p_w[0, email.indices]
                p_e_knowing_r_s = np.power(p_e_knowing_r_s, counts)
                log_p_e_knowing_r_s[i] = np.sum(np.log(p_e_knowing_r_s))

                if (first_email):
                    # RECIPIENT LIKELIHOOD
                    p_r = np.sum([e['freq'] for e in edges_r], dtype=float)
                    p_r /= np.sum([e['freq'] for e in edges])
                    log_p_r[i] = np.log(p_r)

                    # SENDER LIKELIHOOD
                    p_s_knowing_r = np.sum([e['freq'] for e in edges_s_r], dtype=float)
                    p_s_knowing_r /= np.sum([e['freq'] for e in edges_r])
                    log_p_s_knowing_r[i] = np.log(p_s_knowing_r)

            # Compute scores
            scores = log_p_e_knowing_r_s + log_p_s_knowing_r + log_p_r
            scores_sorted, names_sorted = (list(t) for t in zip(*sorted(zip(scores, recs_names))))

            # Store prediction for current email
            y_pred.loc[-1] = [mid, names_sorted[:10]]
            first_email = False

    return y_pred







