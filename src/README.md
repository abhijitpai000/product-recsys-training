| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_datasets() | Pre-processes raw .csv files | -- | item_rec_sys_data.csv, user_rec_sys_data.csv & customer_id_encoder.pkl | --
| trending | compute_top_ten | Computes weighted average and generates top ten products | -- | top_ten_trending.json | --
| item_rec_sys | train_item_rec_sys() | Trains KNNBasic Algorithm | -- | similar_items_algo.pkl | --
| user_rec_sys | train_user_rec_sys() | Trains SVD Algorithm | -- | user_predictions_algo.pkl | --
