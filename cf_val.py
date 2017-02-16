import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

from get_data import get_data, get_val_data
from evaluation import score


def batch_generator(n, batch_size):
    n_batches = n / batch_size
    start_index = 0
    for i in range(n_batches):
        end_index = start_index + batch_size
        yield start_index, end_index
        start_index = end_index
    if n % batch_size:
        end_index = n
        yield start_index, end_index


# Get data
print 'Reading and preprocessing data'
users, items, interactions, target_users, active_items = get_data()

# Get stuff for validation
interactions_train, interactions_val, target_val, users_val = (
    get_val_data(users, items, interactions))

# Get cold start users
warm_start_users = (set(interactions_train.user_id.unique())
                    .intersection(users_val.index))
cold_start_val = users_val.loc[~users_val.index.isin(warm_start_users), :]

user_item = csr_matrix((np.ones(interactions_train.shape[0]),
                        (interactions_train.user_i, interactions_train.item_i)),
                       shape=(users.shape[0], items.shape[0]))


# Run KNN on user_item matrix for target users
print 'KNN'

n_neighbors = 300

knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
knn.fit(user_item)

column_indices = []
row_indices = []

for start, end in tqdm(batch_generator(users_val.shape[0], 1000)):
    indices = users_val.iloc[start:end, :].user_i.values
    neighbors = knn.kneighbors(user_item[indices, :])

    for i in range(end - start):
        user_i = indices[i]
        if user_i not in cold_start_val.user_i:
            for j in range(n_neighbors):
                cosine = 1 - neighbors[0][i][j]
                user_i_2 = neighbors[1][i][j]
                if cosine > 0 and user_i != user_i_2 and user_i:
                    row_indices.append(i + start)
                    column_indices.append(user_i_2)

user_sim_user = csr_matrix(
    (np.ones(len(row_indices)), (row_indices, column_indices)),
    shape=(users_val.shape[0], users.shape[0]))


# Get aggregaion of similar users items and remove non active
print 'aggregation'

user_sim_item = user_sim_user.dot(user_item)

active_user_sim_item = user_sim_item.tocsc()[:, active_items.item_i.values].tocsr()

# Make resulting dataframe
print 'Making resulting dataframe'

res = []
for i in range(active_user_sim_item.shape[0]):
    recommended = active_user_sim_item[i].indices[active_user_sim_item[i].data.argsort()[-30:]]
    recommended = active_items.index[recommended].values
    res.append(recommended)

df = pd.DataFrame({'recommended': res}, index=users_val.index)

# Scoring
print 'Scoring'

intersect = np.array([len(set(a).intersection(b)) for a, b in zip(df.recommended, target_val.relevant)])
print 'Number of intersections = {}'.format(intersect.sum())
s = score(df, target_val)
print 'Score = {}'.format(s)
print ('Leaderbord score = {}'
       .format(s / users_val.shape[0] * 50000))
print ('Full score = {}'
       .format(s / users_val.shape[0] * 150000))

# Save result
print 'Saving'

df.recommended = df.recommended.apply(lambda a: ','.join(a.astype(str)))
df.to_csv('../own_data/cf_val_full.csv', sep='\t')
