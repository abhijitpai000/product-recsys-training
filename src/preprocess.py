"""
Performs data pre-processing.
- Combines dataset
- Returns item_rec_sys_data.csv
- Returns user_rec_sys_data.csv
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import joblib
import random


def _make_item_rec_sys_data(review_data):
    """
    Generates item_rec_sys_data by obfuscating customer ids.

    Parameters
    ----------
        review_data: combined_data.

    Yields
    ------
        item_rec_sys_data.csv

    Returns
    -------
        item_rec_sys_data.

    """
    item_rec_sys_data = review_data.copy()

    # Encoding Customer Ids.
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(item_rec_sys_data[["customer_id"]])
    item_rec_sys_data["customer_id"] = encoded.astype("int").copy()

    # saving encoder.
    file_path = Path.cwd() / "models/customer_id_encoder.pkl"
    joblib.dump(encoder, file_path)

    # saving dataset.
    file_path = Path.cwd() / "datasets/item_rec_sys_data.csv"
    item_rec_sys_data.to_csv(file_path, index=False)

    return item_rec_sys_data


def _make_user_rec_sys_data(item_rec_sys_data):
    """
    Generates user_rec_sys_data.csv by random sampling data of 100 customers.

    Parameters
    ----------
        item_rec_sys_data: item_rec_sys_data.

    Yields
    ------
        user_rec_sys_data.csv

    Returns
    -------
        user_rec_sys_data.

    """

    # Extracting customers with at least 2 reviews.
    review_counts = item_rec_sys_data.groupby("customer_id").count()[["review_score"]]
    all_ids = list(review_counts[review_counts["review_score"] > 1].index)

    # Random Sample 100 ids.
    random_sample_ids = random.choices(all_ids, k=100)
    user_rec_sys_data = item_rec_sys_data[item_rec_sys_data["customer_id"].isin(random_sample_ids)]

    # Resetting Index.
    user_rec_sys_data.reset_index(inplace=True)
    user_rec_sys_data = user_rec_sys_data.drop(["customer_id", "index"], axis=1)
    user_rec_sys_data.reset_index(inplace=True)
    user_rec_sys_data = user_rec_sys_data.rename(columns={"index": "customer_id"})

    # Saving file.
    file_path = Path.cwd() / "datasets/user_rec_sys_data.csv"
    user_rec_sys_data.to_csv(file_path, index=False)

    return user_rec_sys_data


def make_datasets():
    """
    Combines raw .csv files into a structured dataframe.

    Yields
    ------
        item_rec_sys_data.csv
        user_rec_sys_data.csv
    """
    customers = pd.read_csv("datasets/olist_customers_dataset.csv")
    orders = pd.read_csv("datasets/olist_orders_dataset.csv")
    reviews = pd.read_csv("datasets/olist_order_reviews_dataset.csv")
    items = pd.read_csv("datasets/olist_order_items_dataset.csv")
    products = pd.read_csv("datasets/olist_products_dataset.csv")
    translation = pd.read_csv("datasets/product_category_name_translation.csv")

    # Customers + Order Details
    data = customers[["customer_unique_id", "customer_id"]].copy()
    data = data.merge(orders[["order_id", "order_purchase_timestamp", "customer_id"]], on="customer_id", how="left")

    # Data + Order Reviews.
    data = data.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")

    # Data + Product ID from items.
    data = data.merge(items[["order_id", "product_id"]], on="order_id", how="left")

    # Product Category Translations.
    products = products.merge(translation[["product_category_name", "product_category_name_english"]],
                              on="product_category_name", how="left")

    # Data + Product Categories.
    data = data.merge(products[["product_id", "product_category_name_english"]], on="product_id", how="left")

    # Drop Duplicates.
    data.drop_duplicates(inplace=True)

    # Review data raw
    review_data = data[["customer_unique_id", "product_category_name_english", "review_score"]].copy()
    review_data["product_category_name_english"].fillna("Others", axis=0, inplace=True)  # Fix Missing categories

    # Re-naming columns
    review_data.rename(columns={"customer_unique_id": "customer_id",
                                "product_category_name_english": "product_category"},
                       inplace=True)

    # Creating datasets for recommendation systems.
    item_rec_sys_data = _make_item_rec_sys_data(review_data)
    user_rec_sys_data = _make_user_rec_sys_data(item_rec_sys_data)
    return
