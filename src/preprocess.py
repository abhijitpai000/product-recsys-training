"""
Performs data pre-processing.
- Combines dataset
- Encodes "Customer ID" and "Product Category"
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path
import joblib


def _combine_datasets():
    """
    Combines raw .csv files into a structured dataframe.

    Returns
    -------
        review_data_raw dataframe
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
    review_data_raw = data[["customer_unique_id", "product_category_name_english", "review_score"]].copy()
    review_data_raw["product_category_name_english"].fillna("Others", axis=0, inplace=True)  # Fix Missing categories
    return review_data_raw


def _encode_categorical(review_data_raw):
    """
    Encoding Customer ID and Product Category.

    Parameters
    ----------
        review_data_raw: pandas dataframe.

    Yields
    ------
        ord_encoder.pkl in "models"

    Returns
    -------
        review_data pandas dataframe
    """
    ord_encoder = OrdinalEncoder()
    cat_cols = ["customer_unique_id", "product_category_name_english"]
    encoded = pd.DataFrame(ord_encoder.fit_transform(review_data_raw[cat_cols]),
                           columns=["customer_id", "product_category"],
                           index=review_data_raw[cat_cols].index)

    review_data = encoded.join(review_data_raw["review_score"])

    file_path = Path.cwd() / "datasets/review_data.csv"
    review_data.to_csv(file_path, index=False)

    encoder_path = Path.cwd() / "models/ord_encoder.pkl"
    joblib.dump(ord_encoder, encoder_path)
    return review_data


def make_dataset():
    """
    Performs pre-processing
    - Combines datasets
    - Encodes categorical features

    Yields
    ------
        review_data.csv in "datasets"
        encoder.pkl in "models"
    Returns
    -------
        review_data
    """
    review_data_raw = _combine_datasets()
    review_data = _encode_categorical(review_data_raw=review_data_raw)
    return review_data
