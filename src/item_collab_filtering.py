"""
Train KNNBasic Item-Collaborative Filtering algorithm for finding similar items.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise import dump


def train_model():
    """
    Trains KNNBasic Model.

    Yields
    ------
        item_colab_filter.pkl

    Returns
    -------
        #todo: Add returns.
    """
    review_data = pd.read_csv("datasets/review_data.csv")

    # Creating Data object.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df=review_data, reader=reader)
    trainset = data.build_full_trainset()

    # Training Algorithm.
    np.random.seed(0)
    sim_options = {
        "name": "cosine",
        "user_based": False
    }
    algo = KNNBasic(k=10, sim_options=sim_options, verbose=False)
    algo.fit(trainset)

    # Saving Algorithm.
    file_path = Path.cwd() / "models/similar_items_algo.pkl"
    dump.dump(file_path, algo=algo)
    return
