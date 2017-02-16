import numpy as np
from tqdm import trange


# main scoring def:
def score(S, T):
    score = 0.0
    assert np.array_equal(S.index.values, T.index.values)
    for i in trange(T.shape[0]):
        r = S.iloc[i, 0]
        t = T.iloc[i, 0]
        score += (
            20 * (precition_at_k(r, t, 2) + precition_at_k(r, t, 4) +
                  recall(r, t) + user_success(r, t)) +
            10 * (precition_at_k(r, t, 6) + precition_at_k(r, t, 20)))

    return score


def precition_at_k(recommended, relevant, k):
    return len(set(recommended[:k]).intersection(relevant)) / float(k)


# recall = fraction of relevant, retrieved items (30 items
# are allowed to be submitted at maximum per user):
def recall(recommended, relevant):
    if relevant:
        return (len(set(recommended[:30]).intersection(relevant)) /
                float(len(relevant)))
    else:
        return 0.0


# user success = was at least one relevant item recommended for a given user?
def user_success(recommended, relevant):
    if len(set(recommended[:30]).intersection(relevant)):
        return 1.0
    else:
        return 0.0
