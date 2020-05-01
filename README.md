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

## `tutorial.md`

`tutorial.md` contains examples on how this code can be implemented from
beginning to finish via the command line and python implementation. It
also recommends how to test the code for functionality given the long
processing times of feature extraction

## `test.py`
Example usage: 

    $ python test.py

`test.py` is a script to test the functionality of the python code, and
is meant for anybody who wants to evaluate the functionality of this
code to be able to easily do so. Please follow instructions on setting
up the conda environment as detailed in `tutorial.md` before running. 

By default, `test.py` runs with sample.tsv and dev_test_sample.tsv, two
sub-datasets of train.tsv and dev_test.tsv respectively. `test.py` also
includes code for running all of the code from start to finish, starting
with reviews.csv, the raw data file available on
[Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews). For more
information on providing other files, as well as command line and python
implementation, please see `tutorial.md`. 

## `data_util.py`
Example usage: 

    $ python data_util.py --data_path "./data/reviews.csv"

`data_util.py` contains functions for reading and processing review data
from the kaggle dataset, and can be modified with relative ease to work
with other datasets. 

Running this script from the command line as shown will take the
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

    $ python get_features.py --data_path "./data/sample.tsv" --name "sample"
    ... 
    $ python get_features.py --data_path "./data/dev_test_sample.tsv" --name "dev_test_sample"

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

    $ cd data
    $ python ../ablation.py --train_file "sample.tsv" --train_feature_array "sample.npy" \
    --test_file "dev_test_sample.tsv" --test_feature_array "dev_test_sample.npy"

`ablation.py` contains functions for creating different conditions
through feature ablation, as well as functions to fit and predict those
conditions with sklearn `LogisticRegression()` instances

Running this script from the command line as shown will create 5
different train and test conditions:

1. Review Text Features
2. Review Text + Review Readability Features
3. Review Text + Readability + Meta
4. Review Text + Readability + Summary
5. All Features

The script will then train and test a Logistic Regression model on said
conditions and predict the output for the test set. The models are
scored for Accuracy, Precision, Recall, and F1, the outputs of which are
saved to a log file in the script's directory. The log file also
contains the hyperparameters used, which by default are:

- Scale: False (whether or not to scale the data prior to training)
- minimum_votes: 10.0 (minimum number of votes required to be included)
- help_boundary: 0.6 (cutoff point for if a review is considered
  helpful)


### Functions:

- `scale_data()`: scales data with
  `sklearn.preprocessing.StandardScaler` instance
- `create_ablation_sets()`: Takes a full feature array with matching
  data_dict of ReviewerData instance as input and returns 5 sets of
  features as listed above
- `run_logreg()`: Fits logistic regression model for given condition and
  returns fitted model
- `get_zero_rule()`: For gold standard tags array y, returns array of
  length len(y) with zero rule tags, that is the most common tag as the
  prediction every time
- `score_model()`: scores models with `sklearn.metrics`
- `ablation()`: main routine; parses arguments, loads files and fits all
  models for the given conditions, creates the log file and saves to
  `ablation.py`'s directory 

## directory: `data`

Contains data for reviews, including the following samples for testing
the code:

- `sample.tsv`: first 1000 lines of train.tsv
- `dev_test_sample.tsv`: first 500 lines of dev_test.tsv

Files ending in `.tsv` are tabular files containing data found in the
Kaggle dataset. Files ending in `.npy` are the corresponding feature
arrays extracted with get_features.HelpfulnessVectorizer