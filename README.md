# Helpfulness Score Prediction with Python

**Ryan A. Mannion**

**Georgetown University**

Final Project for LING472 - Computational Linguistics
with Advanced Python

Spring 2020

---
The purpose of this project is to predict the helpfulness scores of
reviews left on online marketplaces. Given a review and certain other
metadata, can we engineer features that allow us to accurately predict a
review's helpfulness without the need for users to vote? 

## `data_util.py`
Example usage: 

    $ python data_util.py --data_path "./data/reviews.csv"

`data_util.py` contains functions for reading and processing review data
from the kaggle dataset, and can be modified with relative ease to work
with other datasets. 

##### Functions:

- `read_data()`: Reads data from csv/tsv file and returns a dictionary
  with column name as keys and cell contents as values
- `data_to_tsv()`: Outputs data from dictionary output of read_data to
  tsv, allows for selection of certain columns
- `remove_html()`: Removes select html from an input text
- `filter_data()`: Takes a full feature array and filters it based on
  hyperparameters (e.g. minimum number of votes, helpfulness cutoff)
  
##### Classes:
- `ReviewerData`: Combines the above functions into a class which
  handles the loading and splitting of data into dev and test sets
  
  - Attributes: 
    
    - `data_file`: (str) path to loaded data file
    - `delimiter_type`: (str) delimiter used in `data_file` (i.e. csv or
      tsv)
    - `data_dict`: (dict) dictionary containing data from data_file as
      described in `read_data()`
    - `train`: shuffled subset of data_dict, 80%
    - `dev_test`: shuffled subset of data_dict, 10%
    - `test`: shuffled subset of data_dict, 10%
  - Methods
    - `split_data()`: splits data into dev and test sets 


## `get_features.py`


## `ablation.py`


## directory: `data`


## directory: `data/feature_arrays`