from data_util import ReviewerData
from get_features import HelpfulnessVectorizer
from ablation import ablation

"""
To load, split, and use ~full~ datasets, use this code
(Note: this will take a long time, as the size of the train data is quite large)
"""
# reviews = "./data/reviews.csv"
# reviewer_data = ReviewerData(data_file=reviews, delimiter=',')
# reviewer_data.split_data()
# train_data = reviewer_data.train
# test_data = reviewer_data.dev_test

"""
To load pre-split sample datasets, use this code
    - sample.tsv: first 1000 reviews of train.tsv
    - dev_test_sample.tsv: first 500 reviews of dev_test.tsv
"""
train_data = ReviewerData(data_file="data/sample.tsv", delimiter='\t')
test_data = ReviewerData(data_file="data/dev_test_sample.tsv", delimiter='\t')

# Extract Features
# ----------------
test_vectorizer = HelpfulnessVectorizer(data=test_data)
test_vectors = test_vectorizer.get_features()
train_vectorizer = HelpfulnessVectorizer(data=train_data)
train_vectors = train_vectorizer.get_features()

ablation(train_data, train_vectors, test_data, test_vectors)
