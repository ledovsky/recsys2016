import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import trange

from get_data import get_data, get_val_data
from evaluation import score


# Get data
print 'Reading and preprocessing data'
users, items, interactions, target_users, active_items = get_data()

users.career_level = users.career_level.fillna(3)
users.career_level = users.career_level.replace(0, 3)

# Get stuff for validation
interactions_train, interactions_val, target_val, users_val = (
    get_val_data(users, items, interactions, n=2000, random_state=42))


print 'Count Vectorizer'
count_vec = CountVectorizer(binary=True, max_features=5000)
count_vec.fit(users_val.jobroles)
active_items_title_t = count_vec.transform(active_items.title)
active_items_tags_t = count_vec.transform(active_items.tags)
users_jobroles_t = count_vec.transform(users_val.jobroles)


print 'Getting sparse intersection matrices'
jobroles_title = users_jobroles_t.dot(active_items_title_t.T)
jobroles_tags = users_jobroles_t.dot(active_items_tags_t.T)

res = []

print 'Iterating over users'
for i in trange(users_val.shape[0]):

    user = users_val.iloc[i, :]

    item_scores = pd.Series(0, index=active_items.index)
    # title/job

    intersection = pd.Series(jobroles_title[i, :].toarray()[0], index=active_items.index)
    item_scores.loc[intersection[intersection > 0][:100].index] += intersection[intersection > 0][:100].values

    intersection = pd.Series(jobroles_tags[i, :].toarray()[0], index=active_items.index)
    item_scores.loc[intersection[intersection > 0][:100].index] += intersection[intersection > 0][:100].values

    selection = active_items[(active_items.discipline_id == user.discipline_id) & (active_items.region == user.region)]
    if selection.shape[0] > 100:
        selection = selection.sample(100)
    item_scores.loc[selection.index] += 2

    selection = active_items[(active_items.industry_id == user.industry_id) & (active_items.region == user.region)]
    if selection.shape[0] > 100:
        selection = selection.sample(100)
    item_scores.loc[selection.index] += 1


    selection = active_items[active_items.career_level == user.career_level]
    item_scores = item_scores.loc[selection.index]

    res.append(list(item_scores.sort_values(ascending=False)[:30].index))

df = pd.DataFrame({'recommended': res}, index=users_val.index)

# Scoring

intersect = np.array([len(set(a).intersection(b)) for a, b in zip(df.recommended, target_val.relevant)])
print 'Number of intersections = {}'.format(intersect.sum())
s = score(df, target_val)
print 'Score = {}'.format(s)
print ('Leaderbord score = {}'
       .format(s / users_val.shape[0] * 50000))
print ('Full score = {}'
       .format(s / users_val.shape[0] * 150000))

# Save result

df.recommended = df.recommended.apply(lambda a: ','.join(a.astype(str)))
df.to_csv('../own_data/baseline_val.csv', sep='\t')
