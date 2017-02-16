import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm

from get_data import get_data


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

target_users = users.loc[target_users.user_id]

# Get cold start users
warm_start_users = (set(interactions.user_id.unique())
                    .intersection(target_users.index))
cold_start = target_users.loc[~target_users.index.isin(warm_start_users), :]

user_item = csr_matrix((np.ones(interactions.shape[0]),
                        (interactions.user_i, interactions.item_i)),
                       shape=(users.shape[0], items.shape[0]))


# Run KNN on user_item matrix for target users
print 'KNN'

n_neighbors = 300

knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
knn.fit(user_item)

column_indices = []
row_indices = []

for start, end in tqdm(batch_generator(target_users.shape[0], 1000)):
    indices = target_users.iloc[start:end, :].user_i.values
    neighbors = knn.kneighbors(user_item[indices, :])

    for i in range(end - start):
        user_i = indices[i]
        if user_i not in cold_start.user_i:
            for j in range(n_neighbors):
                cosine = 1 - neighbors[0][i][j]
                user_i_2 = neighbors[1][i][j]
                if cosine > 0 and user_i != user_i_2 and user_i:
                    row_indices.append(i + start)
                    column_indices.append(user_i_2)

user_sim_user = csr_matrix(
    (np.ones(len(row_indices)), (row_indices, column_indices)),
    shape=(target_users.shape[0], users.shape[0]))


# Get aggregaion of similar users items and remove non active
print 'aggregation'

user_sim_item = user_sim_user.dot(user_item).tocsc()
del user_sim_user

active_user_sim_item = user_sim_item.tocsc()[:, active_items.item_i.values].tocsr()


# Make resulting dataframe
print 'Making resulting dataframe'

res = []
for i in range(active_user_sim_item.shape[0]):
    recommended = active_user_sim_item[i].indices[active_user_sim_item[i].data.argsort()[-30:]]
    recommended = active_items.index[recommended].values
    res.append(recommended)

df = pd.DataFrame({'recommended': res}, index=target_users.index)

# Save result

df.recommended = df.recommended.apply(lambda a: ','.join(a.astype(str)))
df.to_csv('../own_data/cf_full_submission.csv', sep='\t')
