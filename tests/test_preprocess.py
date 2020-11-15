"""Test Pre-process module"""

import pytest
import pandas as pd


@pytest.fixture
def load_datasets():
    """
    Loads item_rec_sys_data.csv and user_rec_sys_data.csv
    """
    item_rec_data = pd.read_csv("../datasets/item_rec_sys_data.csv")
    user_rec_data = pd.read_csv("../datasets/user_rec_sys_data.csv")
    return item_rec_data, user_rec_data


def test_item_rec_data(load_datasets):
    """
    Test Shape of item_rec_sys_data.csv
    """
    assert load_datasets[0].shape == (103435, 3)


def test_user_rec_data(load_datasets):
    """
    Test Shape of user_rec_sys_data.csv
    """
    assert load_datasets[1].shape == (221, 3)