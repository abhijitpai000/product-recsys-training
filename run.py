"""Generate Trained algorithms through command line."""

from src.preprocess import make_datasets
from src.trending import compute_top_ten
from src.item_rec_sys import train_item_rec_sys
from src.user_rec_sys import train_user_rec_sys


if __name__ == '__main__':
    make_datasets()
    compute_top_ten()
    train_item_rec_sys()
    train_user_rec_sys()
