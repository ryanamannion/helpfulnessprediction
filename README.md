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

Running this script from the command line as shows will take the
argument of --data_path and save three sub-datasets of that file: train,
dev_test, and test. These files are saved as a .tsv with a \t delimiter

### Functions:

- `read_data()`: Reads data from csv/tsv file and returns a dictionary
  with column name as keys and cell contents as values
- `data_to_tsv()`: Outputs data from dictionary output of read_data to
  tsv, allows for selection of certain columns
- `remove_html()`: Removes select html from an input text
- `filter_data()`: Takes a full feature array and filters it based on
  hyperparameters (e.g. minimum number of votes, helpfulness cutoff)
  
### Classes:
- `ReviewerData`: Combines the above functions into a class which
  handles the loading and splitting of data into dev and test sets
  
  - Attributes: 
    - `data_file`: (str) path to loaded data file
    - `delimiter_type`: (str) delimiter used in `data_file` (i.e. csv or
      tsv)
    - `data_dict`: (dict) dictionary containing data from data_file as
      described in `read_data()`
    - `train`: shuffled subset of `data_dict`, 80%
    - `dev_test`: shuffled subset of `data_dict`, 10%
    - `test`: shuffled subset of `data_dict`, 10%
  - Methods:
    - `split_data()`: splits data into dev and test sets 


## `get_features.py`
Example usage: 

    $ python get_features.py --data_path "./data/sample.tsv" --name "sample_features"

`get_features.py` contains a class HelpfulnessVectorizer for extracting
features from a set of texts. Running from the command line as shown
above will extract features from the given file and save the numpy
ndarray to a binary .npy file with the name given in --name

`get_features.py` uses the library `spaCy` to process the text and
compute features. 18 features total are extracted from 4 groups: Summary
Features, Meta Feature(s), Simple Review Features, and Review
Readability Features. The following is the order of features in the full
`numpy` feature array

###### Summary Text Features:

0. Number of sentences
1.  Number of tokens
2.  Number of characters
3.  Average tokens/ sentence
4.  Average characters/ token 

###### Meta Features:

5.  Star rating of five 

###### Review Text Features:

6.  Number of sentences
7.  Number of tokens
8.  Number of characters
9.  Average tokens/ sentence
10.  Average characters/ token

###### Readability Features (Review text Only):

11.  Flesch-Kincaid Grade Level
12.  Flesch-Kincaid Reading Ease
13.  Dale Chall
14.  SMOG
15.  Coleman-Liau Index
16.  Automated Readability Index
17.  FORCAST
  
### Classes:
- `HelpfulnessVectorizer`
  
  - Attributes: 
    - `spacy_model`: spacy language model to be used for processing
      text, deafaults to `en_web_core_lg`
    - `nlp`: `spacy` object after loading `spacy_model`
    - `read`: `spacy_readability.Readability` (dependency) instance
    - `data`: (dict) data to be vectorized, in the format of
      `ReviewerData.data_dict`
    - `feature_array`: (ndarray) None upon initialization, but is filled
      with completed feature array when complete
  - Methods:
    - `get_features()`: extracts 18 features and returns a numpy
      ndarray, also fills the attribute `feature_array`. Can be easily
      saved to a file with np.save()

## `ablation.py`
Example usage: 

    $ python ablation.py --train_file "path_to_file" --train_feature_array "path_to_file" --test_file "path_to_file" --test_feature_array "path_to_file"

`data_util.py` contains functions for reading and processing review data
from the kaggle dataset, and can be modified with relative ease to work
with other datasets. 

Running this script from the command line as shows will take the
argument of --data_path and save three sub-datasets of that file: train,
dev_test, and test. These files are saved as a .tsv with a \t delimiter

### Functions:

- `read_data()`: Reads data from csv/tsv file and returns a dictionary
  with column name as keys and cell contents as values
- `data_to_tsv()`: Outputs data from dictionary output of read_data to
  tsv, allows for selection of certain columns
- `remove_html()`: Removes select html from an input text
- `filter_data()`: Takes a full feature array and filters it based on
  hyperparameters (e.g. minimum number of votes, helpfulness cutoff)
  
### Classes:
- `ReviewerData`: Combines the above functions into a class which
  handles the loading and splitting of data into dev and test sets
  
  - Attributes: 
    - `data_file`: (str) path to loaded data file
    - `delimiter_type`: (str) delimiter used in `data_file` (i.e. csv or
      tsv)
    - `data_dict`: (dict) dictionary containing data from data_file as
      described in `read_data()`
    - `train`: shuffled subset of `data_dict`, 80%
    - `dev_test`: shuffled subset of `data_dict`, 10%
    - `test`: shuffled subset of `data_dict`, 10%
  - Methods:
    - `split_data()`: splits data into dev and test sets 



## directory: `data`


## directory: `data/feature_arrays`