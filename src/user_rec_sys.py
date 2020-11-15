"""
Trains User Recommendation system.

"""
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.dump import dump
from pathlib import Path


def train_user_rec_sys():
    """
    Trains SlopeOne Algorithm for generating predictions for all users.

    Yields
    ------
        user_predictions_algo.pkl
    """
    user_rec_sys_data = pd.read_csv("datasets/user_rec_sys_data.csv")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(user_rec_sys_data, reader)
    trainset = data.build_full_trainset()

    # Training.
    algo = SVD()
    algo.fit(trainset)

    # Saving Algorithm.
    file_path = Path.cwd() / "models/user_predictions_algo.pkl"
    dump(file_path, algo=algo)
    return
