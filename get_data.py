import datetime

import pandas as pd
import numpy as np


def get_data():

    # Read raw data
    users = pd.read_csv('../raw_data/users.csv', delimiter='\t', index_col=0)
    items = pd.read_csv('../raw_data/items.csv', delimiter='\t', index_col=0)
    interactions = pd.read_csv('../raw_data/interactions.csv', delimiter='\t')
    target_users = pd.read_csv('../raw_data/target_users.csv')

    # Extract data
    interactions.created_at = pd.to_datetime(interactions.created_at, unit='s')
    interactions['date'] = interactions.created_at.apply(lambda d: d.date())

    # Process text fields
    items.title = items.title.fillna('')
    items.tags = items.tags.fillna('')
    users.jobroles = users.jobroles.fillna('')
    items.loc[:, 'title'] = items.title.apply(lambda s: s.replace(',', ' '))
    items.loc[:, 'tags'] = items.tags.apply(lambda s: s.replace(',', ' '))
    users.loc[:, 'jobroles'] = users.jobroles.apply(lambda s: s.replace(',', ' '))

    # Add integer indices
    users['user_i'] = np.arange(users.shape[0])
    items['item_i'] = np.arange(items.shape[0])

    # Add indices to interactions dataframe
    interactions = interactions.merge(users[['user_i']], left_on='user_id',
                                      right_index=True, how='inner')
    interactions = interactions.merge(items[['item_i']], left_on='item_id',
                                      right_index=True, how='inner')

    # Get active items
    active_items = items[items.active_during_test == 1]

    return users, items, interactions, target_users, active_items


def get_val_data(users, items, interactions, n=None, random_state=42):


    # We use last 7 days for validation as recsys organizers offered


    # Split interactions

    interactions_train = interactions[interactions.date <
                                      datetime.date(2015, 11, 2)]

    active_items = items[items.active_during_test == 1]
    interactions_val = interactions[
        (interactions.date >= datetime.date(2015, 11, 2)) &
        (interactions.item_id.isin(active_items.index))]


    # Create target

    target_val = (interactions_val
                  .groupby('user_id')['item_id']
                  .agg({'relevant': lambda a: list(np.unique(a))}))

    # Get subsample if necessary
    if n is not None:
        target_val = target_val.sample(n, random_state=random_state)

    target_val.sort_index(inplace=True)

    # Select users for recommendation
    users_val = users.loc[target_val.index]

    return interactions_train, interactions_val, target_val, users_val
