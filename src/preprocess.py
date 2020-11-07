"""
Performs data pre-processing.
- Combines dataset
- Encodes "Customer ID" and "Product Category"
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import joblib


def _combine_datasets():
    """
    Combines raw .csv files into a structured dataframe.

    Returns
    -------
        review_data dataframe
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

    # Encoding Customer Ids.
    encoder = OrdinalEncoder()
    encoded = encoder.fit_transform(review_data[["customer_id"]])
    review_data["customer_id"] = encoded.astype("int").copy()

    # saving encoder.
    file_path = Path.cwd() / "datasets/customer_id_encoder.pkl"
    joblib.dump(encoder, file_path)
    return review_data


def make_dataset():
    """
    Performs pre-processing
    - Combines datasets

    Yields
    ------
        review_data.csv in "datasets"
    Returns
    -------
        review_data
    """
    review_data = _combine_datasets()

    file_path = Path.cwd() / "datasets/review_data.csv"
    review_data.to_csv(file_path, index=False)
    return review_data
