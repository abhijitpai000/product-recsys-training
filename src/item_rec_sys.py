"""
Train KNNBasic Item-Collaborative Filtering algorithm for finding similar items.
"""

import pandas as pd
from pathlib import Path
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise import dump
import json


def _compute_inner_item_ids(item_rec_sys_data, algo, trainset):
    """
    Computes inner items ids using raw ids.

    Parameters
    ----------
        item_rec_sys_data: pandas dataframe.
        algo: Fit Algorithm.
        trainset: Trainset data object.

    Yields
    ------
        item_inner_id_mapping.json
    """
    # Raw IDs.
    product_categories = list(item_rec_sys_data["product_category"].value_counts().index)

    # Mapping Inner Ids.
    item_inner_id_mapping = {}

    for category in product_categories:
        inner_id = algo.trainset.to_inner_iid(category)
        item_inner_id_mapping[category] = inner_id

    # Saving .json file.
    file_path = Path.cwd() / "datasets/item_id_mapping.json"
    json.dump(item_inner_id_mapping, open(file_path, 'w'))
    return


def train_item_rec_sys():
    """
    Trains KNNBasic Model.

    Yields
    ------
        similar_items_algo.pkl
    """
    item_rec_sys_data = pd.read_csv("datasets/item_rec_sys_data.csv")

    # Creating Data object.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df=item_rec_sys_data, reader=reader)
    trainset = data.build_full_trainset()

    # Training Algorithm.
    sim_options = {
        "name": "cosine",
        "user_based": False
    }
    algo = KNNBasic(k=10, sim_options=sim_options, verbose=False)
    algo.fit(trainset)

    # Extract inner id mappings.
    _compute_inner_item_ids(item_rec_sys_data, algo=algo, trainset=trainset)

    # Saving Algorithm.
    file_path = Path.cwd() / "models/similar_items_algo.pkl"
    dump.dump(file_path, algo=algo)
    return
