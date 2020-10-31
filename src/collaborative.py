""" Collaborative Filtering """

from surprise import Dataset, Reader
import pandas as pd

def create_data_object():
    """
    Create data object from pandas dataframe
    """
    review_data = pd.read_csv("datasets/review_data.csv")

    # Defining reader object.
    reader = Reader(line_format='user item rating',
                    rating_scale=(1, 5))
    data = Dataset(review_data, reader=reader)
    return
