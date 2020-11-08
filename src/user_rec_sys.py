"""
Trains User Recommendation system.

"""
import pandas as pd
from surprise import SlopeOne
from surprise import Dataset, Reader
from surprise.dump import dump
from pathlib import Path


def _make_user_predictions(trainset, algo):
    """
    Make Predictions for all users.

    Parameters
    ----------
        trainset: trainset data object.
        algo: fit SlopeOne Algorithm.

    Yields
    ------
        all_user_predictions.csv
    """
    testset = trainset.build_anti_testset()
    all_user_predictions = algo.test(testset)
    all_user_predictions = pd.DataFrame(all_user_predictions).drop(["r_ui", "details"], axis=1)
    all_user_predictions = all_user_predictions.rename(columns={"uid": "user_inner_id",
                                                                "est": "estimated_rating",
                                                                "iid": "item_inner_id"})
    # Saving file.
    file_path = Path.cwd() / "datasets/all_user_predictions.csv"
    all_user_predictions.to_csv(file_path, index=False)

    return


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
    algo = SlopeOne()
    algo.fit(trainset)
    _make_user_predictions(trainset, algo)

    # Saving Algorithm.
    file_path = Path.cwd() / "user_predictions_algo.pkl"
    dump(file_path, algo=algo)
    return
