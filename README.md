# Product-Recsys - Model Training
This repository holds the model training part of the product-recsys application from the [product-recsys](https://github.com/abhijitpai000/product-recsys) repo.

## Repository Structure
    .
    ├── datasets                   # .csv file raw and generated during model training.
    ├── models                     # .pkl files generated during model training.
    ├── src                        # Source files, scripts for training.
        ├── preprocess.py                  # Combines .csv files and generates datasets for training.
        ├── trending.py                    # Computes top ten trending products.
        ├── item_rec_sys.py                # Trains KNNBasic algorithm.
        └── user_rec_sys.py                # Trains SVD algorithm.
    ├── tests
        └── test_preprocess.py             # Checks .csv file generated during pre-processing.
    ├── run.py                     # Run the entire training process from command line.
    ├── requirements.txt           # Python packages required for training.
    ├── LICENSE
    └── README.md


## Setup instructions

#### Creating Python environment

This repository has been tested on Python 3.7.6.

1. Cloning the repository:

`git clone https://github.com/abhijitpai000/product-recsys-training.git`

2. Navigate to the git clone repository.

`cd product-recsys-training`

3. Download raw data from the data source link and place in "datasets" directory

4. Install [virtualenv](https://pypi.org/project/virtualenv/)

`pip install virtualenv`

`virtualenv recsys`

5. Activate it by running:

`recsys/Scripts/activate`

6. Install project requirements by using:

`pip install -r requirements.txt`

## Training

1. Download datasets zip file from [Brazilian E-Commerce Public Dataset](https://www.kaggle.com/olistbr/brazilian-ecommerce) and place it in '../datasets' directory, unzip the .zip file (this should produce 9 .csv files)
2. Set the current working directory to 'product-recsys-training' clone directory.

### Train models - Terminal
1. Use run.py to preprocess, train and save files. 

`python run.py`

2. Run tests for a quick sanity check.

`cd tests`

`python test_preprocess.py`

*OR*

### Train models in a Jupyter notebook

Utilize modules from 'src' as python package and run it in Jupyter Notebooks by importing following local modules from the 'src' directory of this repo.

| Module | Function | Description | Parameters | Yields | Returns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| preprocess | make_datasets() | Pre-processes raw .csv files | -- | item_rec_sys_data.csv, user_rec_sys_data.csv & customer_id_encoder.pkl | --
| trending | compute_top_ten() | Computes weighted average and generates top ten products | -- | top_ten_trending.json | --
| item_rec_sys | train_item_rec_sys() | Trains KNNBasic Algorithm | -- | similar_items_algo.pkl | --
| user_rec_sys | train_user_rec_sys() | Trains SVD Algorithm | -- | user_predictions_algo.pkl | --

```python
# Local Imports.
from src.preprocess import make_datasets
from src.trending import compute_top_ten
from src.item_rec_sys import train_item_rec_sys
from src.user_rec_sys import train_user_rec_sys
```

```python
# Combine .csv files and pre-process data.
  make_datasets()
  
# Compute top ten trending.
  compute_top_ten()
  
# Train KNNBasic algorithm.
  train_item_rec_sys()
  
# Train SVD Algorithm.
  train_user_rec_sys()
```
