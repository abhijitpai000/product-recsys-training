"""
Computes Weighted Average for each product.

Formula:
WR = (v/v+m)*R+(m/v+m)*C

v = Total Reviews
m = Minimum votes required to qualify to trending list
R = Average Rating
C = Mean rating across the report.
"""
import pandas as pd
from pathlib import Path


def compute_top_ten():
    """
    Computes Weighted Average for each product and returns top ten.

    Formula:
    WR = (v/v+m)*R+(m/v+m)*C

    v = Total Reviews
    m = Minimum votes required to qualify to trending list
    R = Average Rating
    C = Mean rating across the report.
    """
    review_data = pd.read_csv("datasets/item_rec_sys_data.csv")

    trending = review_data.groupby("product_category")[["review_score"]].agg(["mean", "count"])
    trending.columns = ["avg_review_score", "total_reviews"]

    # Computing C.
    c = trending["avg_review_score"].mean()

    # Computing m, a product should have votes > 90% of items.
    m = trending["total_reviews"].quantile(0.9)

    # Weighted Average.
    trending["weighted_average"] = (trending["total_reviews"] / (trending["total_reviews"] + m)) * \
                                   (trending["avg_review_score"] + (m / (trending["total_reviews"] + m)) * c)

    # Top 10.
    top_ten = trending.sort_values("weighted_average", ascending=False).head(10)
    top_ten = top_ten.drop("avg_review_score", axis=1)
    top_ten.reset_index(inplace=True)

    file_path = Path.cwd() / "datasets/top_ten_trending.json"
    top_ten.to_json(file_path)
    return
